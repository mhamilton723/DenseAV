import glob
import os
from abc import ABC, abstractmethod
from glob import glob
from os.path import join
from pathlib import Path
from typing import List, Set

import audioread
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchaudio
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset, DataLoader, default_collate, Subset, ConcatDataset
from tqdm import tqdm

from denseav.constants import AUDIO_MASK, AUDIO_POS_MASK, IMAGE_MASK, IMAGE_INPUT
from denseav.data.make_tarballs import untar_all
from denseav.shared import norm, prep_waveform


def sample_choice(choices, probs):
    # Check that probabilities sum to 1 and are non-negative
    assert sum(probs) == 1, "Probabilities must sum to 1"
    assert all(p >= 0 for p in probs), "Probabilities cannot be negative"

    # Convert probs to a tensor
    probs_tensor = torch.tensor(probs)

    # Sample a choice according to the probabilities
    index = torch.multinomial(probs_tensor, 1).item()

    # Return the sampled choice
    return choices[index]


def grid_frames(frames):
    top_row = torch.cat([frames[0], frames[1]], dim=2)
    bottom_row = torch.cat([frames[2], frames[3]], dim=2)
    return torch.cat([top_row, bottom_row], dim=3)


def create_mixed_image(pos_frame, neg_frame, patch_size):
    # Step 1: Check that patch_size evenly divides the image dimensions
    b, c, h, w = pos_frame.shape
    assert h % patch_size == 0 and w % patch_size == 0, "Patch size must evenly divide image dimensions"

    # Step 2: Create a random binary mask with the same number of patches as the image
    mask = torch.randint(0, 2, (b, 1, h // patch_size, w // patch_size))

    # Step 3: Create a new image using patches from pos_frame and neg_frame according to the mask
    # Upscale the mask to the size of the image
    mask_upscaled = F.interpolate(mask.to(torch.float32), scale_factor=patch_size)

    # Use the mask to create a mixed frame
    mixed_frame = mask_upscaled * pos_frame + (1 - mask_upscaled) * neg_frame

    return mixed_frame, mask_upscaled


class AVDataset(ABC, Dataset):

    @abstractmethod
    def _dataset_folder(self) -> str:
        pass

    @abstractmethod
    def _load_info(self, split) -> pd.DataFrame:
        """
        This function should return a dataframe with at least a column "id"
        @return:
        """
        pass

    @abstractmethod
    def _missing_threshold(self) -> float:
        pass

    @abstractmethod
    def default_target_length(self) -> int:
        pass

    def target_length(self):
        if self.override_target_length is not None:
            return self.override_target_length
        else:
            return self.default_target_length()

    def _frame_root(self) -> str:
        return join(self.root, "frames", self.split)

    def _video_root(self) -> str:
        return join(self.root, "videos", self.split)

    def _audio_root(self) -> str:
        return join(self.root, "audio", self.split)

    def _semseg_root(self) -> str:
        return join(self.root, "annotations", self.split)

    def _embed_root(self) -> str:
        return join(self.root, "embedding", self.audio_embed_model, self.split)

    def _label_root(self) -> str:
        return join(self.root, "pseudo-labels")

    def _hn_root(self) -> str:
        return join(self.root, "hard_negatives")

    def _all_video_files(self) -> Set[str]:
        return set(str(p) for p in Path(join(self._video_root())).rglob('*'))

    def _all_frame_files(self) -> Set[str]:
        return set(str(p) for p in Path(join(self._frame_root())).rglob('*'))

    def _all_audio_files(self) -> Set[str]:
        return set(str(p) for p in Path(join(self._audio_root())).rglob('*'))

    def _all_embed_files(self) -> Set[str]:
        return set(str(p) for p in Path(join(self._embed_root())).rglob('*'))

    def _get_frame_files(self, row) -> List[str]:
        return [self._frame_root() + "/" + row["id"] + f"_{i}.jpg" for i in range(self._expected_num_frames())]

    def _get_semseg_file(self, row) -> str:
        raise NotImplementedError("Class has not implemented _get_semseg_files")

    def _get_audio_file(self, row) -> str:
        return self._audio_root() + "/" + row["id"] + ".mp3"

    def _get_video_file(self, row) -> str:
        return self._video_root() + "/" + row["id"] + ".mp4"

    def _get_embed_file(self, row) -> str:
        return self._embed_root() + "/" + row["id"] + ".npz"

    def _add_files_to_metadata(self, df) -> pd.DataFrame:
        tqdm.pandas()

        if self.use_audio_embed:
            df["embed_file"] = df.progress_apply(self._get_embed_file, axis=1)

        if self.use_audio or self.use_spec:
            df["audio_file"] = df.progress_apply(self._get_audio_file, axis=1)

        if self.use_frames:
            df["frame_files"] = df.progress_apply(self._get_frame_files, axis=1)

        if self.use_semseg:
            df["semseg_file"] = df.progress_apply(self._get_semseg_file, axis=1)

        df = self._filter_valid_metadata(df)

        if self.use_hn:
            loaded = np.load(join(self._hn_root(), "original", f"{self.split}_hard_negatives.npz"))
            df["hn0"] = [t for t in torch.tensor(loaded["indices_0"])]
            df["hn1"] = [t for t in torch.tensor(loaded["indices_1"])]

        return df

    def _split_name(self, split):
        return split

    def _filter_valid_metadata(self, df: pd.DataFrame) -> pd.DataFrame:

        print("MY_DIR ", list(glob(join(self.root, "*"))))
        if self.use_audio_embed:
            missing_embed_files = set(df['embed_file']) - self.all_embed_files
            valid_audio = ~df['embed_file'].isin(missing_embed_files)
            print("ALL EMBED ", len(self.all_embed_files))
        elif self.use_audio or self.use_spec:
            missing_audio_files = set(df['audio_file']) - self.all_audio_files
            valid_audio = ~df['audio_file'].isin(missing_audio_files)
            print("ALL AUDIO ", len(self.all_audio_files))

        if self.use_frames:
            missing_frame_files = set(
                item for sublist in df['frame_files'].tolist() for item in sublist) - self.all_frame_files
            valid_frames = df['frame_files'].apply(lambda x: not any(file in missing_frame_files for file in x))
            print("ALL FRAMES ", len(self.all_frame_files))
            df["is_valid"] = valid_audio & valid_frames
        else:
            df["is_valid"] = valid_audio

        percent_missing = (1 - (df["is_valid"].sum() / len(df)))

        assert percent_missing <= self._missing_threshold(), \
            f"Too many missing files: %{round(percent_missing * 100.0, 2)}"
        assert len(df) > 0, "No files found"
        return df[df["is_valid"]]

    def __init__(
            self,
            root: str,
            split: str = "train",
            use_frames=False,
            frame_transform=None,
            use_audio=False,
            use_spec=False,
            use_audio_embed=False,
            use_hn=False,
            use_caption=False,
            use_semseg=False,
            neg_audio=False,
            use_davenet_spec=False,
            use_fnac_spec=False,
            n_label_frames=196,
            label_transform=None,
            audio_embed_model="hubert",
            n_frames=1,
            audio_transform=None,
            audio_aug=False,
            spec_transform=None,
            spec_mel_bins=128,
            spec_mean=-6.6268077,
            spec_std=5.358466,
            sample_rate=16000,
            override_target_length=None,
            use_tags=False,
            extra_audio_masking=False,
            audio_level=False,
            quad_mixup=0.0,
            bg_mixup=0.0,
            patch_mixup=0.0,
            patch_size=8,
    ):
        super(AVDataset).__init__()
        self.pytorch_data_dir = root
        self.split = self._split_name(split)
        self.root = join(root, self._dataset_folder())
        self.use_frames = use_frames
        self.frame_transform = frame_transform
        self.use_audio = use_audio
        self.use_spec = use_spec
        self.use_audio_embed = use_audio_embed
        self.use_davenet_spec = use_davenet_spec
        self.use_fnac_spec = use_fnac_spec
        self.use_hn = use_hn
        self.use_caption = use_caption
        self.label_transform = label_transform
        self.audio_embed_model = audio_embed_model
        self.audio_aug = audio_aug
        self.n_frames = n_frames
        self.audio_transform = audio_transform
        self.spec_transform = spec_transform
        self.spec_mel_bins = spec_mel_bins
        self.spec_mean = spec_mean
        self.spec_std = spec_std
        self.use_semseg = use_semseg
        self.override_target_length = override_target_length
        self.use_tags = use_tags
        self.extra_audio_masking = extra_audio_masking
        self.neg_audio = neg_audio
        self.audio_level = audio_level

        self.quad_mixup = quad_mixup
        self.bg_mixup = bg_mixup
        self.patch_mixup = patch_mixup
        self.patch_size = patch_size

        self.sample_rate = sample_rate
        self.n_label_frames = n_label_frames

        if self.use_audio_embed:
            self.all_embed_files = self._all_embed_files()

        if self.use_audio or self.use_spec:
            self.all_audio_files = self._all_audio_files()

        if self.use_frames:
            self.all_frame_files = self._all_frame_files()

        self.metadata = self._add_files_to_metadata(self._load_info(self.split))

        assert len(self.metadata) > 0

    def __len__(self):
        return len(self.metadata)

    @abstractmethod
    def _expected_num_frames(self) -> int:
        pass

    def get_audio_mask(self, real_length, padded_length, target_size):
        if not isinstance(real_length, torch.Tensor):
            real_length = torch.tensor(real_length)
            padded_length = torch.tensor(padded_length)

        n_frames = ((real_length / padded_length) * target_size).to(torch.int64)
        oh = F.one_hot(n_frames, num_classes=target_size + 1)
        if len(oh.shape) == 1:
            oh = oh.unsqueeze(0)
        return (1 - torch.cumsum(oh, dim=1))[:, :-1].to(torch.bool)

    def _base_get_item(self, item):
        id = self.metadata["id"].iloc[item]
        data_dict = {"metadata": {"id": id, "index": item}}

        if self.use_tags and "tags" in self.metadata:
            tags = torch.tensor(self.metadata["tags"].iloc[item])
            tag_oh = torch.zeros(self.num_tags, dtype=torch.float32)
            tag_oh[tags] += 1
            data_dict["tags"] = tag_oh

        if self.use_audio or self.use_spec:
            audio_file = self.metadata["audio_file"].iloc[item]
            data_dict["metadata"]["audio_file"] = audio_file
            loaded_waveform, obs_sr = torchaudio.load(audio_file)
            loaded_waveform = loaded_waveform[0]

            if self.neg_audio:
                neg_audio_file = self.metadata["audio_file"].iloc[torch.randint(0, len(self), size=(1,)).item()]
                data_dict["metadata"]["neg_audio_file"] = neg_audio_file
                neg_waveform, neg_obs_sr = torchaudio.load(neg_audio_file)
                neg_waveform = neg_waveform[0]
            else:
                neg_waveform, neg_obs_sr = None, None

            (waveform,
             spectrogram,
             audio_length,
             total_length,
             original_length,
             mask,
             pos_mask) = prep_waveform(
                loaded_waveform,
                obs_sr,
                self.target_length(),
                self.spec_mel_bins,
                self.spec_mean,
                self.spec_std,
                self.sample_rate,
                self.use_spec,
                False,
                self.extra_audio_masking,
                neg_waveform,
                neg_obs_sr,
                self.audio_level,
                self.audio_aug
            )

            if self.spec_transform is not None and spectrogram is not None:
                spectrogram = self.spec_transform(spectrogram)

            if self.audio_transform is not None:
                waveform = self.audio_transform(waveform)

            data_dict["audio"] = waveform
            data_dict[AUDIO_MASK] = mask
            data_dict[AUDIO_POS_MASK] = pos_mask
            data_dict["audio_length"] = audio_length
            data_dict["original_length"] = original_length
            data_dict["total_length"] = total_length
            if spectrogram is not None:
                data_dict["spec"] = spectrogram

            if mask.mean() < .04:
                return None

        if self.use_davenet_spec:
            from data.DavenetUtilities import davenet_load_audio
            audio_file = self.metadata["audio_file"].iloc[item]
            spec, n_frames = davenet_load_audio(audio_file)
            data_dict["davenet_spec"] = spec

        if self.use_fnac_spec:
            from featurizers.FNACAVL import load_spectrogram as fnac_load_spectrogram
            audio_file = self.metadata["audio_file"].iloc[item]
            data_dict["fnac_spec"] = fnac_load_spectrogram(audio_file, 3)

        if self.use_audio_embed:
            loaded = np.load(self.metadata["embed_file"].iloc[item])
            data_dict["audio_emb"] = loaded["feat"]
            data_dict["audio_length"] = loaded["audio_length"]
            data_dict["total_length"] = loaded["total_length"]
            data_dict["original_length"] = loaded["original_length"]
            data_dict[AUDIO_MASK] = self.get_audio_mask(
                data_dict["audio_length"],
                data_dict["total_length"],
                data_dict["audio_emb"].shape[-1]) \
                .squeeze().to(torch.float32)
            data_dict[AUDIO_POS_MASK] = data_dict[AUDIO_MASK].to(torch.float32)

        if self.use_frames:

            def get_frames(item):
                file_group = self.metadata["frame_files"].iloc[item]
                if self.n_frames is not None:
                    selected_frames = torch.randperm(len(file_group))[:self.n_frames]
                    file_group = [file_group[i] for i in selected_frames]
                data_dict["metadata"]["frame_files"] = file_group
                images = [Image.open(file).convert("RGB") for file in file_group]

                if self.frame_transform is not None:
                    images = torch.cat([self.frame_transform(img).unsqueeze(0) for img in images], dim=0)

                return images, file_group

            no_mixup = 1.0 - (self.bg_mixup + self.quad_mixup + self.patch_mixup)

            mixup_type = sample_choice(
                ["quad", "bg", "patch", None],
                [self.quad_mixup, self.bg_mixup, self.patch_mixup, no_mixup]
            )

            if mixup_type == "quad":
                indices = [item] + torch.randint(0, len(self), size=(3,)).numpy().tolist()
                frames_and_files = [get_frames(i) for i in indices]
                file_group = frames_and_files[0][1]
                perm = torch.randperm(4)
                all_frames = [F.interpolate(frames_and_files[i][0], scale_factor=0.5, mode="bilinear") for i in
                              perm]
                b, c, h, w = all_frames[0].shape
                indices = [indices[p] for p in perm]
                masks = [(torch.ones(b, 1, h, w) if index == item else torch.zeros(b, 1, h, w)) for index in
                         indices]

                data_dict[IMAGE_INPUT] = grid_frames(all_frames)
                data_dict[IMAGE_MASK] = grid_frames(masks)
            elif mixup_type == "bg":
                neg_item = torch.randint(0, len(self), size=(1,)).item()
                neg_frame, _ = get_frames(neg_item)
                pos_frame, file_group = get_frames(item)

                b, c, h, w = neg_frame.shape
                neg_mask = torch.zeros(b, 1, h, w)
                pos_mask = torch.ones(b, 1, h, w)

                if torch.rand(1).item() > 0.5:
                    bg_frame = neg_frame
                    bg_mask = neg_mask
                    fg_frame = F.interpolate(pos_frame, scale_factor=0.5, mode="bilinear")
                    fg_mask = F.interpolate(pos_mask, scale_factor=0.5, mode="bilinear")
                else:
                    bg_frame = pos_frame
                    bg_mask = pos_mask
                    fg_frame = F.interpolate(neg_frame, scale_factor=0.5, mode="bilinear")
                    fg_mask = F.interpolate(neg_mask, scale_factor=0.5, mode="bilinear")

                start_h = torch.randint(0, h // 2, size=(1,))
                start_w = torch.randint(0, w // 2, size=(1,))
                bg_frame[:, :, start_h:start_h + fg_frame.shape[2], start_w:start_w + fg_frame.shape[3]] = fg_frame
                bg_mask[:, :, start_h:start_h + fg_frame.shape[2], start_w:start_w + fg_frame.shape[3]] = fg_mask

                data_dict["frames"] = bg_frame
                data_dict["image_masks"] = bg_mask

            elif mixup_type == "patch":
                neg_item = torch.randint(0, len(self), size=(1,)).item()
                neg_frame, _ = get_frames(neg_item)
                pos_frame, file_group = get_frames(item)
                frames, masks = create_mixed_image(pos_frame, neg_frame, self.patch_size)
                data_dict["frames"] = frames
                data_dict["image_masks"] = masks

            elif mixup_type is None:
                frames, file_group = get_frames(item)

                data_dict["frames"] = frames
                b, c, h, w = frames.shape
                data_dict["image_masks"] = torch.ones(b, 1, h, w)
            else:
                raise ValueError(f"Unknown mixup type {mixup_type}")

            if "original_length" in data_dict:
                if self._expected_num_frames() == 1:
                    frame_nums = torch.tensor([0])
                else:
                    frame_nums = torch.tensor([
                        int(f.split("/")[-1].split("_")[-1].split(".")[0]) for f in file_group])

                data_dict["frame_nums"] = frame_nums
                frame_fracs = ((frame_nums + .5) / (self._expected_num_frames()))
                frame_position = (frame_fracs * data_dict["original_length"]) / data_dict["total_length"]
                data_dict["frame_position"] = frame_position

        if self.use_caption:
            if "word" in self.metadata:
                words = self.metadata["word"].iloc[item]
                start = self.metadata["start"].iloc[item]
                end = self.metadata["end"].iloc[item]
                if isinstance(words, float):
                    words = [""]
                    start = [0.0]
                    end = [-1.0]

                data_dict["caption"] = {
                    "words": words,
                    "start": start,
                    "end": end,
                }
            if "text" in self.metadata:
                data_dict["text"] = self.metadata["text"].iloc[item]

        if self.use_semseg:
            semseg_path = join(self._semseg_root(), self.metadata["semseg_file"].iloc[item])
            semseg = Image.open(semseg_path)
            if self.label_transform is not None:
                semseg = np.array(self.label_transform(semseg))
            data_dict["semseg"] = semseg
            data_dict["metadata"]["semseg_file"] = semseg_path

            # if hasattr(self, "num_classes"):
            #     data_dict["num_pixels_per_class"] = F.one_hot(
            #         torch.tensor(semseg).to(torch.int64), self.num_classes() + 1).sum(dim=[0, 1])

        return data_dict

    def __getitem__(self, item):
        try:
            data_dict = self._base_get_item(item)
            if self.use_hn:
                indices = torch.cat([self.metadata["hn0"].iloc[item], self.metadata["hn1"].iloc[item]], dim=0)
                neg_index = indices[torch.randint(0, indices.shape[0], (1,))]
                negative_dict = self._base_get_item(neg_index)
                data_dict["negatives"] = negative_dict
            return data_dict
        except (audioread.exceptions.NoBackendError, EOFError) as e:
            # raise e
            bad_path = self.metadata["audio_file"].iloc[item]
            print(e)
            print(f"Removing bad audio file {bad_path}")
            # os.remove(bad_path)
            return None
        except ValueError as e:
            # raise e
            bad_path = self.metadata["audio_file"].iloc[item]
            if "Input signal length=0" in str(e):
                print(e)
                print(f"Removing bad file {bad_path} due to input signal length=0")
            #     os.remove(bad_path)
            return None
        except OSError as e:
            # raise e
            bad_paths = self.metadata["frame_files"].iloc[item]
            for bad_path in bad_paths:
                print(e)
                print(f"Removing bad frame file {bad_path}")
            return None
        except RuntimeError as e:
            # raise e
            bad_path = self.metadata["audio_file"].iloc[item]
            print(e)
            print(f"Removing bad audio file {bad_path}")
            # os.remove(bad_path)
            return None


class PlacesAudio(AVDataset):

    def _load_info(self, split) -> pd.DataFrame:
        df = pd.read_json(join(os.path.dirname(self._audio_root()), "metadata", f"{split}.json"))
        df["id"] = df["data"].apply(lambda d: d["wav"][5:-4])

        if self.use_caption:
            if split == "train":
                word_df = pd.read_json(
                    join(os.path.dirname(self._audio_root()), "metadata", f"word-alignment-{split}.json")
                )
            else:
                word_df = pd.read_csv(
                    join(os.path.dirname(self._audio_root()), "metadata", f"word-alignment-{split}.csv")) \
                    .groupby("id").aggregate(lambda g: list(g)).reset_index().drop("Unnamed: 0", axis=1)
            df = pd.merge(df, word_df, on="id", how="outer")
        return df

    def _missing_threshold(self) -> float:
        # return 0.0
        return 0.97  # TODO fix

    def _expected_num_frames(self):
        return 1

    def default_target_length(self) -> int:
        return 20

    def _frame_root(self) -> str:
        return join(os.path.dirname(self.root), "places_subset")

    def _audio_root(self) -> str:
        return join(self.root, "wavs")

    def _embed_root(self) -> str:
        return join(self.root, "embedding", self.audio_embed_model)

    def _dataset_folder(self) -> str:
        return "PlacesAudio_400k_distro"

    def _get_audio_file(self, row) -> str:
        return join(self._audio_root(), row["id"] + ".wav")

    def _get_frame_files(self, row) -> List[str]:
        return [join(self._frame_root(), row["data"]["image"])]

    def _get_embed_file(self, row) -> str:
        return join(self._embed_root(), row["id"] + ".npz")


class AudioSet(AVDataset):
    def _expected_num_frames(self):
        return 10

    def default_target_length(self) -> int:
        return 20

    def _dataset_folder(self) -> str:
        return "audioset-raw"

    def _missing_threshold(self) -> float:
        if self.split == "val" or self.split == "test":
            return 0.02
        else:
            return 0.17

    def train_seg_file(self):
        return "unbalanced_train_segments.csv"

    def _load_info(self, split) -> pd.DataFrame:
        if split == "train":
            df = pd.read_csv(join(self.root, "metadata", self.train_seg_file()))
        elif split == "val" or split == "test":
            df = pd.read_csv(join(self.root, "metadata", "eval_segments_subset.csv"))
        else:
            raise ValueError(f"Unknown split {split}")

        labels = pd.read_csv(join(self.root, "metadata", "class_labels_indices.csv"))
        mid_to_index = dict(zip(labels["mid"], labels["index"]))
        df["tags"] = df["positive_labels"].apply(lambda l: [mid_to_index[e] for e in l.strip('"').split(",")])

        self.num_tags = max(*[i for k, i in mid_to_index.items()]) + 1
        df["id"] = df.apply(lambda r: f"{r.YTID}_{r.start_seconds}_{r.end_seconds}", axis=1)
        return df

    def _frame_root(self) -> str:
        return join(self.root, "frames")

    def _audio_root(self) -> str:
        return join(self.root, "audio")

    def _all_frame_files(self) -> Set[str]:
        frame_files = set()

        for entry in os.scandir(self._frame_root()):
            if entry.is_file():
                frame_files.add(entry.path)
            elif entry.is_dir():
                for subentry in os.scandir(entry.path):
                    if subentry.is_file():
                        frame_files.add(subentry.path)

        return frame_files

    def _all_audio_files(self) -> Set[str]:
        return set(entry.path for entry in os.scandir(self._audio_root()) if entry.is_file())

    def _all_embed_files(self) -> Set[str]:
        return set(entry.path for entry in os.scandir(self._embed_root()) if entry.is_file())

    def _embed_root(self) -> str:
        return join(self.root, "embedding", self.audio_embed_model)

    def prefix(self):
        return ""

    def _get_audio_file(self, row) -> str:
        return f"{self.root}/audio/{self.prefix()}{row.id}.mp3"

    def _get_frame_files(self, row) -> List[str]:
        return [f"{self.root}/frames/frame_{fn}/{self.prefix()}{row.id}.jpg" for fn in range(10)]

    def _get_embed_file(self, row) -> str:
        return f"{self.root}/embedding/{self.audio_embed_model}/{self.prefix()}{row.id}.npz"


class AudioSetEval(AudioSet):

    def _dataset_folder(self) -> str:
        return "audioset-eval"

    def _get_frame_files(self, row) -> List[str]:
        base_path = f"{self.root}/frames/{self.prefix()}{row.id}_"
        return [base_path + f"{fn}.jpg" for fn in range(10)]

    def prefix(self):
        return ""


class ADE20K(AVDataset):

    def _split_name(self, split):
        if split == "val":
            return "validation"
        elif split == "train":
            return "training"
        else:
            raise ValueError(f"Unknown split name {split}")

    def _load_info(self, split) -> pd.DataFrame:
        df = pd.read_json(join(self.root, "metadata_with_caption_dedup.json"))
        df["id"] = df["image"]
        df = df[df["image"].apply(lambda f: f.split("/")[0] == split)]

        if self.use_caption:
            df["word"] = df["caption"].apply(lambda c: c["words"])
            df["start"] = df["caption"].apply(lambda c: c["start"])
            df["end"] = df["caption"].apply(lambda c: c["end"])
            df["text"] = df["word"].apply(lambda l: " ".join(l))
        return df

    def _missing_threshold(self) -> float:
        return 0.03

    def _expected_num_frames(self):
        return 1

    def default_target_length(self) -> int:
        return 20

    def _dataset_folder(self) -> str:
        return "ADE20K"

    def _frame_root(self) -> str:
        return join(self.root, "frames")

    def _audio_root(self) -> str:
        return join(self.root, "audio")

    def _semseg_root(self) -> str:
        return join(self.root, "annotations")

    def _embed_root(self) -> str:
        return join(self.root, "embedding", self.audio_embed_model)

    def _get_audio_file(self, row) -> str:
        return join(self._audio_root(), row["audio"])

    def _get_frame_files(self, row) -> List[str]:
        return [join(self._frame_root(), row["image"])]

    def _get_semseg_file(self, row) -> str:
        return join(self._semseg_root(), row["seg"])

    def _get_embed_file(self, row) -> str:
        return join(self._embed_root(), row["image"].replace(".jpg", ".npz"))

    def num_classes(self):
        return 3662


class ADE20KPromptedBase(AVDataset):

    def _expected_num_frames(self):
        return 1

    def default_target_length(self) -> int:
        return 20

    def _frame_root(self) -> str:
        return join(self.root, "frames")

    def _audio_root(self) -> str:
        return join(self.root, "audio")

    def _semseg_root(self) -> str:
        return join(self.root, "annotations")

    def _embed_root(self) -> str:
        return join(self.root, "embedding", self.audio_embed_model)

    def _get_frame_files(self, row) -> List[str]:
        return [join(self._frame_root(), row["image_location"])]

    def _get_semseg_file(self, row) -> str:
        return join(self._semseg_root(), row["image_location"].replace(".jpg", "_seg.png"))

    def _get_embed_file(self, row) -> str:
        return join(self._embed_root(), row["image_location"].replace(".jpg", ".npz"))

    def num_classes(self):
        return 3662

    def _missing_threshold(self) -> float:
        return 0.0


class ADE20KSpeechPrompted(ADE20KPromptedBase):

    def _get_audio_file(self, row) -> str:
        return join(self._audio_root(), row["speech_prompt_file"].split("/")[-1])

    def _dataset_folder(self) -> str:
        return "ADE20KSpeechPrompted"

    def _audio_root(self) -> str:
        # return join(self.root, "audio-noise-10") # TODO Remove
        return join(self.root, "audio")  # TODO Remove

    def _load_info(self, split) -> pd.DataFrame:
        df = pd.read_csv(join(self.root, "prompted_segmentation.csv"))
        df = df[df["speech_prompt_file"].apply(lambda s: isinstance(s, str))]
        df = df[df["ade_class_id"].apply(lambda id: id != 0)]
        df["id"] = df["image_location"]
        return df


class ADE20KSoundPrompted(ADE20KPromptedBase):

    def _get_audio_file(self, row) -> str:
        return join(self._audio_root(), row["vggsound_file"].split("/")[-1])

    def _dataset_folder(self) -> str:
        return "ADE20KSoundPrompted"

    def _load_info(self, split) -> pd.DataFrame:
        df = pd.read_csv(join(self.root, "prompted_segmentation.csv"))
        df = df[df["vggsound_file"].apply(lambda s: isinstance(s, str))]
        df = df[df["ade_class_id"].apply(lambda id: id != 0)]
        df["id"] = df["image_location"]
        return df


class PlacesAndAudioSet(Dataset):

    def __init__(self, **kwargs):
        self.ds1 = PlacesAudio(**kwargs, n_frames=1)
        self.ds2 = AudioSet(**kwargs, n_frames=1)

    def __len__(self):
        return len(self.ds1)

    def __getitem__(self, item):
        if torch.rand(1).item() > .5:
            d = self.ds2[torch.randint(0, len(self.ds2) - 1, size=(1,)).item()]
            if d is not None:
                d["source"] = 1
        else:
            d = self.ds1[item]
            if d is not None:
                d["source"] = 0
        return d


class AVDataModule(pl.LightningDataModule):
    def __init__(self,
                 dataset_name,
                 load_size,
                 image_aug,
                 audio_aug,
                 extra_audio_masking,
                 audio_model_type,
                 pytorch_data_dir,
                 use_cached_embs,
                 batch_size,
                 num_workers,
                 audio_level,
                 neg_audio,
                 data_for_plotting,
                 use_original_val_set,
                 use_extra_val_sets,
                 quad_mixup,
                 bg_mixup,
                 patch_mixup,
                 patch_size,
                 **kwargs):

        super().__init__()
        self.dataset_name = dataset_name
        self.load_size = load_size
        self.image_aug = image_aug
        self.audio_aug = audio_aug
        self.extra_audio_masking = extra_audio_masking
        self.audio_model_type = audio_model_type
        self.pytorch_data_dir = pytorch_data_dir
        self.use_cached_embs = use_cached_embs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_for_plotting = data_for_plotting
        self.audio_level = audio_level
        self.neg_audio = neg_audio

        self.quad_mixup = quad_mixup
        self.bg_mixup = bg_mixup
        self.patch_mixup = patch_mixup
        self.patch_size = patch_size

        self.loader_args = dict(
            num_workers=self.num_workers,
            batch_size=self.batch_size,
        )
        self.save_hyperparameters()
        self.extra_args = kwargs

        self.use_original_val_set = use_original_val_set
        self.use_extra_val_sets = use_extra_val_sets

    def maybe_unpack(self, remove_source):
        targets = [
            (
                join(self.pytorch_data_dir, "audioset-subset", "frame_archives"),
                join(self.pytorch_data_dir, "audioset-subset", "frames"),
                1
            ),
            (
                join(self.pytorch_data_dir, "audioset-raw", "frame_archives"),
                join(self.pytorch_data_dir, "audioset-raw", "frames"),
                4
            ),
            (
                join(self.pytorch_data_dir, "audioset-raw", "audio_archives"),
                join(self.pytorch_data_dir, "audioset-raw", "audio"),
                1
            ),

        ]

        for (archive_dir, target_dir, n_parts) in targets:
            if not os.path.exists(target_dir) and os.path.exists(archive_dir):
                print(f"Could not find {target_dir}, attempting to unpack archives")
                if os.path.exists(archive_dir):
                    untar_all(archive_dir, target_dir, remove_source)
                else:
                    raise RuntimeError(f"Could not find archive folder: {archive_dir}")

    def get_dataset_by_name(self, name, stage, data_for_plotting, n_frames=None):

        if name == "vggss":
            resize_op = T.Resize((self.load_size, self.load_size), Image.BILINEAR)
        else:
            resize_op = T.Resize(self.load_size, Image.BILINEAR)

        img_transform = T.Compose([
            resize_op,
            T.CenterCrop(self.load_size),
            T.ToTensor(),
            norm])

        if self.image_aug:
            train_img_transform = T.Compose([
                T.RandomResizedCrop(self.load_size),
                T.RandomHorizontalFlip(),
                T.ColorJitter(.2, .2, .2, .2),
                T.RandomGrayscale(),
                T.ToTensor(),
                norm])
            val_img_transform = img_transform
        else:
            train_img_transform = img_transform
            val_img_transform = img_transform

        if self.audio_aug:
            train_audio_aug = True
            val_audio_aug = False
        else:
            train_audio_aug = False
            val_audio_aug = False

        if self.audio_model_type == "hubert":
            from featurizers.Hubert import HubertAudioTransform
            audio_transform = HubertAudioTransform()
        else:
            audio_transform = None

        if self.audio_model_type == "passt":
            sample_rate = 32000
        else:
            sample_rate = 16000

        if not self.use_cached_embs:
            if self.audio_model_type == "hubert":
                self.extra_args["use_audio"] = True
            elif self.audio_model_type in {"audiomae", "audiomae-finetuned", "cavmae", "cavmae-mixed", "imagebind"}:
                self.extra_args["use_spec"] = True
            elif self.audio_model_type == "davenet":
                self.extra_args["use_audio"] = True
                self.extra_args["use_davenet_spec"] = True
            elif self.audio_model_type == "fnac":
                self.extra_args["use_audio"] = True
                self.extra_args["use_fnac_spec"] = True
            else:
                raise ValueError(f"Unknown audio model type {self.audio_model_type}")

            if self.audio_model_type == "cavmae" or self.audio_model_type == "cavmae-mixed":
                self.extra_args["spec_mean"] = -5.081
                self.extra_args["spec_std"] = 4.4849
            elif self.audio_model_type == "imagebind":
                self.extra_args["spec_mean"] = -4.268
                self.extra_args["spec_std"] = 9.138

        # if self.audio_model_type in {"audiomae", "audiomae-finetune", "cavmae"} \
        #         and "override_target_length" not in self.extra_args:
        if "override_target_length" not in self.extra_args:
            self.extra_args["override_target_length"] = 10

        data_args = dict(
            root=self.pytorch_data_dir,
            use_frames=True,
            audio_transform=audio_transform,
            sample_rate=sample_rate,
            audio_level=self.audio_level,
            **self.extra_args
        )

        if n_frames is not None:
            data_args["n_frames"] = n_frames

        train_args = dict(
            frame_transform=train_img_transform,
            extra_audio_masking=self.extra_audio_masking,
            neg_audio=self.neg_audio,
            quad_mixup=self.quad_mixup,
            bg_mixup=self.bg_mixup,
            patch_mixup=self.patch_mixup,
            patch_size=self.patch_size,
            audio_aug=train_audio_aug
        )
        val_args = dict(
            frame_transform=val_img_transform,
            audio_aug=val_audio_aug
        )

        if data_for_plotting:
            val_args["use_audio"] = True
            val_args["use_spec"] = True

        if "ade" in name:
            label_transform = T.Compose([
                T.Resize(self.load_size, Image.NEAREST),
                T.CenterCrop(self.load_size),
                prep_ade_label
            ])
        else:
            label_transform = T.Compose([
                T.Resize(self.load_size, Image.NEAREST),
                T.CenterCrop(self.load_size)
            ])

        val_args["use_audio"] = True
        val_args["label_transform"] = label_transform

        if name == "places-audio":
            dataset_constructor = PlacesAudio
        elif name == "mixed-full":
            dataset_constructor = PlacesAndAudioSet
        elif name == "audio-set-full":
            dataset_constructor = AudioSet
        elif name == "audio-set-eval":
            dataset_constructor = AudioSetEval
        elif name == "ade":
            val_args["use_semseg"] = True
            dataset_constructor = ADE20K
        elif name == "ade-speech-prompted":
            val_args["use_semseg"] = True
            dataset_constructor = ADE20KSpeechPrompted
        elif name == "ade-sound-prompted":
            val_args["use_semseg"] = True
            dataset_constructor = ADE20KSoundPrompted
        else:
            raise ValueError(f"Unknown dataset name {name}")

        data_args["use_audio_embed"] = self.use_cached_embs
        data_args["audio_embed_model"] = self.audio_model_type

        if stage == "full":
            val_dataset = dataset_constructor(split="val", **{**data_args, **val_args})
            train_dataset = dataset_constructor(split="train", **{**data_args, **val_args})
            return ConcatDataset([train_dataset, val_dataset])
        elif stage == "fit":
            return dataset_constructor(split="train", **{**data_args, **train_args})
        elif stage == "validate":
            return dataset_constructor(split="val", **{**data_args, **val_args})
        else:
            raise ValueError(f"Unknown stage: {stage}")

    def _maybe_subset(self, dataset, length):
        if len(dataset) > length and self.dataset_name not in {"ade-sound-prompted", "ade-speech-prompted", "vggss"}:
            print("Using a subset of validation data")
            return Subset(dataset, generate_subset(len(dataset), length))
        else:
            print("Not using val subset")
            return dataset

    def _make_val_datasets(self):
        val_sets = []
        if self.use_original_val_set:
            val_sets.append(self._maybe_subset(self.get_dataset_by_name(
                self.dataset_name, "validate", self.data_for_plotting), 1000))

        if self.use_extra_val_sets:
            val_sets.append(self._maybe_subset(self.get_dataset_by_name(
                "places-audio", "validate", self.data_for_plotting), 1000))
            val_sets.append(self._maybe_subset(self.get_dataset_by_name(
                "audio-set-eval", "validate", False, n_frames=1), 1000))
            val_sets.append(self.get_dataset_by_name(
                "ade-speech-prompted", "validate", True))
            val_sets.append(self.get_dataset_by_name(
                "ade-sound-prompted", "validate", self.data_for_plotting))

        return val_sets

    def setup(self, stage: str):
        if stage == "full":
            self.full_dataset = self.get_dataset_by_name(self.dataset_name, stage, self.data_for_plotting)
        elif stage == "fit":
            self.train_dataset = self.get_dataset_by_name(self.dataset_name, stage, self.data_for_plotting)
            self.val_datasets = self._make_val_datasets()
        elif stage == "validate":
            self.val_datasets = self._make_val_datasets()
        else:
            raise ValueError(f"Unknown stage: {stage}")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, **self.loader_args, collate_fn=custom_coallate)

    def subsampled_train_dataloader(self, k=5000):
        if len(self.train_dataset) > k:
            ds = Subset(self.train_dataset, generate_subset(len(self.train_dataset), k))
        else:
            ds = self.train_dataset

        return DataLoader(ds, shuffle=True, **self.loader_args, collate_fn=custom_coallate)

    def val_dataloader(self):
        return [
            DataLoader(dataset, shuffle=False, **self.loader_args, collate_fn=custom_coallate)
            for dataset in self.val_datasets
        ]

    def full_dataloader(self):
        return DataLoader(self.full_dataset, shuffle=False, **self.loader_args, collate_fn=custom_coallate)


def generate_subset(n, batch, seed=0):
    np.random.seed(seed)
    return np.random.permutation(n)[:batch]


def prep_ade_label(img):
    seg = np.array(img)
    class_labels = (seg[:, :, 0] / 10).astype(np.int32) * 256 + (seg[:, :, 1].astype(np.int32))
    return class_labels


def maybe_replace(e, not_none):
    if e is not None:
        return e
    else:
        print("Warning found a None in the dataset indicitive of a loading failure, replacing it with another item")
        return not_none[0]


empty_caption = {
    "words": [],
    "start": [],
    "end": [],
}


def custom_coallate(l):
    if l is None:
        return l

    not_none = [e for e in l if e is not None]
    assert len(not_none) > 0

    l = [maybe_replace(e, not_none) for e in l]

    to_merge = {}

    def pop_or_default(dict, k, default):
        if k in dict:
            return dict.pop(k)
        else:
            print(f"WARNING: Could not find {k}, using {default}")
            return default

    if "caption" in l[0]:
        to_merge["caption"] = [pop_or_default(l[i], "caption", empty_caption) for i in range(len(l))]

    if "text" in l[0]:
        to_merge["text"] = [pop_or_default(l[i], "text", "") for i in range(len(l))]

    result = default_collate(l)

    return {**result, **to_merge}


if __name__ == "__main__":

    from featurizers.Hubert import HubertAudioTransform

    pytorch_data_dir = "/pytorch-data"
    dataset_constructor = PlacesAudio
    split = "val"

    img_transform = T.Compose([
        T.Resize(224, Image.BILINEAR),
        T.CenterCrop(224),
        T.ToTensor(),
        norm])

    video_transform = T.Compose([
        T.Resize(224, Image.BILINEAR),
        T.CenterCrop(224),
        norm])

    label_transform = T.Compose([
        T.Resize(224, Image.NEAREST),
        T.CenterCrop(224)
    ])

    audio_transform = HubertAudioTransform()

    data_args = dict(
        root=pytorch_data_dir,
        frame_transform=img_transform,
        use_frames=True,
        use_spec=True,
        use_audio=True,
        use_caption=False,
        use_semseg=False,
        label_transform=label_transform,
        audio_transform=audio_transform,
        use_audio_embed=False,
        audio_embed_model="audiomae",
        extra_audio_masking=False,
        neg_audio=False,
        override_target_length=10,
        audio_level=False,
        quad_mixup=.3,
        patch_mixup=.3,
        bg_mixup=.3,
    )


    def return_datasets(dataset_constructor, split):
        dataset = dataset_constructor(split=split, **data_args)
        return dataset


    train_ds = return_datasets(dataset_constructor, split)

    print(len(train_ds))
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=False, num_workers=36, collate_fn=custom_coallate)
    for batch in tqdm(train_loader):
        pass
