import random
from collections import defaultdict, deque
from typing import Any

import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchaudio
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset
from torchaudio.functional import resample


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        image2 = torch.clone(image)
        for t, m, s in zip(image2, self.mean, self.std):
            t.mul_(s).add_(m)
        return image2


class SliceDataset(Dataset):

    def __init__(self, ds, start, end):
        self.ds = ds
        self.start = start
        self.end = end

    def __len__(self):
        return self.end - self.start

    def __getitem__(self, item):
        return self.ds[item + self.start]


class SubsetDataset(Dataset):

    def __init__(self, ds, subset):
        self.ds = ds
        self.subset = subset

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, item):
        return self.ds[self.subset[item]]


norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
unnorm = UnNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def crop_to_divisor(x, patch_size):
    if len(x.shape) == 3:
        C, H, W = x.shape
        return x[:, :(patch_size * (H // patch_size)), :(patch_size * (W // patch_size))]
    elif len(x.shape) == 4:
        B, C, H, W = x.shape
        return x[:, :, :(patch_size * (H // patch_size)), :(patch_size * (W // patch_size))]
    else:
        raise ValueError("x should have 3 or 4 dimensions")


def _remove_axes(ax):
    ax.xaxis.set_major_formatter(plt.NullFormatter())
    ax.yaxis.set_major_formatter(plt.NullFormatter())
    ax.set_xticks([])
    ax.set_yticks([])


def remove_axes(axes):
    if len(axes.shape) == 2:
        for ax1 in axes:
            for ax in ax1:
                _remove_axes(ax)
    else:
        for ax in axes:
            _remove_axes(ax)


def get_image_featurizer(name, token_type="key", **kwargs):
    name = name.lower()

    if name == "vit":
        from denseav.featurizers.DINO import DINOFeaturizer
        patch_size = 16
        model = DINOFeaturizer("vit_small_patch16_224", patch_size, token_type)
        dim = 384
    elif name == "dino16":
        from denseav.featurizers.DINO import DINOFeaturizer
        patch_size = 16
        model = DINOFeaturizer("dino_vits16", patch_size, token_type)
        dim = 384
    elif name == "dino8":
        from denseav.featurizers.DINO import DINOFeaturizer
        patch_size = 8
        model = DINOFeaturizer("dino_vits8", patch_size, token_type)
        dim = 384
    elif name == "clip":
        from denseav.featurizers.CLIP import CLIPFeaturizer
        patch_size = 16
        model = CLIPFeaturizer()
        dim = 512
    elif name == "cavmae":
        from denseav.featurizers.CAVMAE import CAVMAEImageFeaturizer
        model = CAVMAEImageFeaturizer(kwargs["output_root"], model=kwargs.get("model"))
        dim = 768
        patch_size = 16
    elif name == "fnac":
        from denseav.featurizers.FNACAVL import FNACImageFeaturizer
        model = FNACImageFeaturizer(kwargs["output_root"], model=kwargs.get("model"))
        dim = 512
        patch_size = 16
    elif name == "imagebind":
        from denseav.featurizers.ImageBind import ImageBindImageFeaturizer
        model = ImageBindImageFeaturizer(kwargs["output_root"], model=kwargs.get("model"))
        dim = 1024
        patch_size = 16
    elif name == "resnet50":
        from torchvision import models
        model = models.resnet50(pretrained=True)
        model = torch.nn.Sequential(*list(model.children())[:-2])
        patch_size = 1
        dim = 2048
    elif name == "davenet":
        from fdenseav.eaturizers.DAVENet import DavenetImageFeaturizer
        model = DavenetImageFeaturizer()
        patch_size = 1
        dim = 1024
    elif name == "dinov2":
        from denseav.featurizers.DINOv2 import DINOv2Featurizer
        model = DINOv2Featurizer()
        patch_size = 14
        dim = 768
    else:
        raise ValueError("unknown model: {}".format(name))
    return model, patch_size, dim


def get_audio_featurizer(name, **kwargs):
    if name == "davenet":
        from denseav.featurizers.DAVENet import DavenetAudioFeaturizer
        model = DavenetAudioFeaturizer()
        dim = 1024
    elif name == "dino8":
        model, _, dim = get_image_featurizer("dino8")
    elif name == "hubert":
        from featurizers.Hubert import Hubert
        model = Hubert()
        dim = 1024
    elif name == "cavmae":
        from denseav.featurizers.CAVMAE import CAVMAEAudioFeaturizer
        model = CAVMAEAudioFeaturizer(kwargs["output_root"], model=kwargs.get("model"))
        dim = 768
    elif name == "imagebind":
        from denseav.featurizers.ImageBind import ImageBindAudioFeaturizer
        model = ImageBindAudioFeaturizer(kwargs["output_root"], model=kwargs.get("model"))
        dim = 1024
    elif name == "audiomae":
        from denseav.featurizers.AudioMAE import AudioMAE
        model = AudioMAE(kwargs["output_root"], False)
        dim = 768
    elif name == "audiomae-finetuned":
        from denseav.featurizers.AudioMAE import AudioMAE
        model = AudioMAE(kwargs["output_root"], True)
        dim = 768
    else:
        raise ValueError("Unknown audio model type")

    return model, dim


def load_img(image_path, transform):
    return transform(Image.open(image_path)).unsqueeze(0)


def pytorch_to_pil(tensor):
    return Image.fromarray((unnorm(tensor).permute(0, 2, 3, 1).cpu() * 255)
                           .clamp(0, 255).to(torch.uint8).detach().numpy()[0])


def _get_random_window(waveform, mask, min_size, max_size):
    effective_size = mask.sum().to(torch.int64)
    if effective_size <= min_size:
        return waveform, mask
    else:
        window_size = min(torch.randint(low=min_size, high=min(effective_size, max_size), size=()), waveform.shape[0])
        if window_size == waveform.shape[0]:
            window_start = 0
        else:
            window_start = torch.randint(low=0, high=effective_size - window_size, size=())

        new_waveform = torch.zeros_like(waveform)
        new_mask = torch.zeros_like(mask)
        new_waveform[window_start:window_start + window_size] = waveform[window_start:window_start + window_size]
        new_mask[window_start:window_start + window_size] = mask[window_start:window_start + window_size]
        return new_waveform, new_mask


def _splice_clips(clip1, clip2, loc, easing_size):
    assert loc >= 0 and loc < len(clip1), "Invalid location"
    assert easing_size > 0 and easing_size <= len(clip2), "Invalid easing size"

    try:
        assert loc + clip2.shape[0] < clip1.shape[0]
    except Exception as e:
        print(loc, clip2.shape[0], clip1.shape[0])
        raise e

    # Split clip1 into three parts: before splice, easing region, after splice
    before_splice = clip1[:loc]
    after_splice = clip1[loc + clip2.shape[0]:]

    # Compute the fading weights for the easing region
    # fade_in_weights = torch.cos(torch.linspace(1, 0, easing_size, device=clip1.device))
    fade_in_weights = 0.5 * (1 + torch.cos(math.pi * torch.linspace(0, 1, easing_size)))
    fade_out_weights = 1 - fade_in_weights

    clip1_ease = torch.cat([
        fade_in_weights,
        torch.zeros(clip2.shape[0] - easing_size * 2),
        fade_out_weights,
    ])

    mask = torch.cat([torch.ones(loc), clip1_ease, torch.ones(clip1.shape[0] - (loc + clip2.shape[0]))])

    # Apply fading weights to clip1 and clip2 within the easing region
    splice = clip1_ease * clip1[loc:loc + clip2.shape[0]] + (1 - clip1_ease) * clip2

    # Concatenate all parts back together
    spliced_clip = torch.cat((before_splice, splice, after_splice))

    return spliced_clip, mask


def _generate_random_subset(waveform, low, high):
    length = len(waveform)

    # If waveform is smaller than low or has zero length, return unmodified
    if length < low or length == 0:
        return waveform

    # Generate random start index within valid range
    start = random.randint(0, length - low)

    # Generate random subset size within valid range
    subset_size = random.randint(low, min(high, length - start))

    # Extract the random subset from the waveform
    subset = waveform[start: start + subset_size]

    return subset


def level_audio(waveform):
    waveform -= waveform.mean()
    waveform /= waveform.abs.max().valus.clamp_min(.0001)
    return waveform


def prep_waveform(waveform,
                  obs_sr,
                  target_length,
                  spec_mel_bins,
                  spec_mean,
                  spec_std,
                  sample_rate,
                  return_spec,
                  random_clip,
                  extra_audio_masking,
                  neg_waveform,
                  neg_obs_sr,
                  audio_level,
                  audio_aug,
                  ):
    if obs_sr != sample_rate:
        waveform = resample(waveform, obs_sr, sample_rate)
        if audio_level:
            waveform = level_audio(waveform)

    if neg_obs_sr is not None and neg_obs_sr != sample_rate:
        neg_waveform = resample(neg_waveform, neg_obs_sr, sample_rate)
        if audio_level:
            neg_waveform = level_audio(neg_waveform)

    if neg_obs_sr is not None:  # and random.random() > .5:
        neg_waveform_clip = _generate_random_subset(neg_waveform, sample_rate, sample_rate * 4)
        if waveform.shape[0] - neg_waveform_clip.shape[0] > 0:
            start = random.randint(0, waveform.shape[0] - neg_waveform_clip.shape[0] - 1)
            easing = max(int(neg_waveform_clip.shape[0] * 1 / 4), sample_rate // 2)
            easing = min(int(neg_waveform_clip.shape[0] * 1 / 2), easing)
            waveform, pos_mask = _splice_clips(waveform, neg_waveform_clip, start, easing_size=easing)
        else:
            waveform, pos_mask = waveform, torch.ones_like(waveform)
    else:
        waveform, pos_mask = waveform, torch.ones_like(waveform)

    mask = torch.ones_like(waveform)
    original_length = waveform.shape[0]

    if target_length == 10:
        target_samples = 164200  # Result is 1024 after spec
    else:
        target_samples = int(target_length * sample_rate)

    padding = target_samples - original_length

    if padding > 0:
        p = torch.nn.ZeroPad2d((0, padding))
        waveform = p(waveform)
        mask = p(mask)
        pos_mask = p(pos_mask)
    else:
        if random_clip:
            start = torch.randint(0, waveform.shape[0] - target_samples, size=())
        else:
            start = 0
        end = start + target_samples
        waveform = waveform[start:end]
        mask = mask[start:end]
        pos_mask = pos_mask[start:end]

    audio_length = min(original_length, target_samples)
    total_length = target_samples

    if extra_audio_masking:
        min_size = sample_rate // 2
        max_size = total_length
        if original_length > min_size and random.random() > .5:
            waveform, mask = _get_random_window(waveform, mask, min_size, max_size)

    if audio_aug:
        import torchaudio_augmentations as AA
        from torchvision.transforms import RandomApply, Compose

        transform = Compose([
            RandomApply([AA.PolarityInversion()], p=0.5),
            RandomApply([AA.Noise(min_snr=0.001, max_snr=0.005)], p=0.2),
            RandomApply([AA.Gain()], p=0.2),
            RandomApply([AA.HighLowPass(sample_rate=sample_rate)], p=0.2),
            RandomApply([AA.PitchShift(n_samples=waveform.shape[-1], sample_rate=sample_rate)], p=0.2),
            RandomApply([AA.Reverb(sample_rate=sample_rate)], p=0.2)
        ])
        waveform = transform(waveform.unsqueeze(0)).squeeze(0)

    if return_spec:
        spectrogram = torchaudio.compliance.kaldi.fbank(
            waveform.unsqueeze(0) - waveform.mean(),
            htk_compat=True,
            sample_frequency=sample_rate,
            use_energy=False,
            window_type='hanning',
            num_mel_bins=spec_mel_bins,
            dither=0.0,
            frame_shift=10)

        spectrogram = ((spectrogram - spec_mean) / spec_std).unsqueeze(0)
    else:
        spectrogram = None

    if mask.mean() < .04:
        print(f"Bad entry: {mask.mean()}")

    return waveform, spectrogram, audio_length, total_length, original_length, mask, pos_mask


class ToTargetTensor(object):
    def __call__(self, target):
        return torch.as_tensor(np.array(target), dtype=torch.int64).unsqueeze(0)


def show_heatmap(ax,
                 image,
                 heatmap,
                 cmap="bwr",
                 color=False,
                 center=False,
                 show_negative=False,
                 cax=None,
                 vmax=None,
                 vmin=None):
    frame = []

    if color:
        frame.append(ax.imshow(image))
    else:
        bw = np.dot(np.array(image)[..., :3] / 255, [0.2989, 0.5870, 0.1140])
        bw = np.ones_like(image) * np.expand_dims(bw, -1)
        frame.append(ax.imshow(bw))

    if center:
        heatmap -= heatmap.mean()

    if not show_negative:
        heatmap = heatmap.clamp_min(0)

    heatmap = F.interpolate(heatmap.unsqueeze(0).unsqueeze(0), (image.shape[0], image.shape[1])) \
        .squeeze(0).squeeze(0)

    if vmax is None:
        vmax = np.abs(heatmap).max()
    if vmin is None:
        vmin = -vmax

    hm = ax.imshow(heatmap, alpha=.5, cmap=cmap, vmax=vmax, vmin=vmin)
    if cax is not None:
        plt.colorbar(hm, cax=cax, orientation='vertical')

    frame.extend([hm])
    return frame


class TorchPCA(object):

    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X):
        self.mean_ = X.mean(dim=0)
        unbiased = X - self.mean_.unsqueeze(0)
        U, S, V = torch.pca_lowrank(unbiased, q=self.n_components, center=False, niter=4)
        self.components_ = V.T
        self.singular_values_ = S
        return self

    def transform(self, X):
        t0 = X - self.mean_.unsqueeze(0)
        projected = t0 @ self.components_.T
        return projected


def pca(image_feats_list, dim=3, fit_pca=None):
    device = image_feats_list[0].device

    def flatten(tensor, target_size=None):
        if target_size is not None and fit_pca is None:
            F.interpolate(tensor, (target_size, target_size), mode="bilinear")
        B, C, H, W = tensor.shape
        return feats.permute(1, 0, 2, 3).reshape(C, B * H * W).permute(1, 0).detach().cpu()

    if len(image_feats_list) > 1 and fit_pca is None:
        target_size = image_feats_list[0].shape[2]
    else:
        target_size = None

    flattened_feats = []
    for feats in image_feats_list:
        flattened_feats.append(flatten(feats, target_size))
    x = torch.cat(flattened_feats, dim=0)

    if fit_pca is None:
        # fit_pca = PCA(n_components=dim, svd_solver='arpack').fit(np.nan_to_num(x.detach().numpy()))
        fit_pca = TorchPCA(n_components=dim).fit(x)

    reduced_feats = []
    for feats in image_feats_list:
        # x_red = torch.from_numpy(fit_pca.transform(flatten(feats)))
        x_red = fit_pca.transform(flatten(feats))
        x_red -= x_red.min(dim=0, keepdim=True).values
        x_red /= x_red.max(dim=0, keepdim=True).values
        B, C, H, W = feats.shape
        reduced_feats.append(x_red.reshape(B, H, W, dim).permute(0, 3, 1, 2).to(device))

    return reduced_feats, fit_pca


def merge_col(fig, axes, col):
    gs = axes[0, col].get_gridspec()
    for ax in axes[:, col]:
        ax.remove()
    return fig.add_subplot(gs[:, col])


def visualize_av_features(
        audio,
        video,
        feat_a,
        feat_v,
        att_a,
        n_frames,
        norm_before_pca=True,
        axes=None,
        fig=None,
        modify_fig=True,
        video_time=0,
        fit_pca=None
):
    assert (len(audio.shape) == 3)  # C, F, T
    assert (len(video.shape) == 4)  # T, C, H, W
    assert (len(feat_a.shape) == 2)  # C, T
    assert (len(feat_v.shape) == 4)  # T, C, H, W
    assert (len(att_a.shape) == 2)  # F, T

    ac, af, at = audio.shape
    fac, fat = feat_a.shape

    if modify_fig:
        if axes is None:
            fig, axes = plt.subplots(3, 3, figsize=(5 * 3, 5))
            fig.tight_layout()

        bigax1 = merge_col(fig, axes, 0)
        bigax2 = merge_col(fig, axes, 1)
        _remove_axes(bigax1)
        _remove_axes(bigax2)
        remove_axes(axes[:, 2])
    else:
        bigax1 = fig.axes[-2]
        bigax2 = fig.axes[-1]

    frame_v = unnorm(video).permute(0, 2, 3, 1).detach().cpu()
    frame_v -= frame_v.min()
    frame_v /= frame_v.max()

    frame_a = audio.detach().cpu()
    frame_a -= frame_a.min()
    frame_a /= frame_a.max()

    if norm_before_pca:
        [red_feat_v], fit_pca = pca([F.normalize(feat_v, dim=1)], fit_pca=fit_pca)
        [red_feat_a], _ = pca([F.normalize(feat_a.unsqueeze(0).unsqueeze(-1), dim=1)], fit_pca=fit_pca)
    else:
        [red_feat_v], fit_pca = pca([feat_v], fit_pca=fit_pca)
        [red_feat_a], _ = pca([feat_a.unsqueeze(0).unsqueeze(-1)], fit_pca=fit_pca)

    red_feat_v = red_feat_v.permute(0, 2, 3, 1).detach().cpu()
    red_feat_a = red_feat_a.permute(0, 2, 3, 1)[0].detach().cpu()

    if red_feat_a.shape[0] == 1:
        new_height = int((frame_a.shape[0] / frame_a.shape[1]) * red_feat_a.shape[1])
        red_feat_a = torch.broadcast_to(
            red_feat_a, (new_height, red_feat_a.shape[1], red_feat_a.shape[2]))
        plt_att_a = torch.broadcast_to(att_a, (new_height, att_a.shape[1]))
    else:
        plt_att_a = att_a

    frac_signal = n_frames / fat
    n_at = int(at * frac_signal)

    return [bigax1.imshow(frame_v[video_time]),
            bigax2.imshow(red_feat_v[video_time]),
            axes[0, 2].imshow(frame_a[:, :n_at]),
            axes[0, 2].set_title("Spectrogram"),
            axes[1, 2].imshow(red_feat_a[:, :n_frames]),
            axes[1, 2].set_title("Audio Features"),
            axes[2, 2].imshow(plt_att_a[:, :n_frames], vmin=0),
            axes[2, 2].set_title("Audio Attention")], fig, fit_pca


def create_label_tensor(labels, starts, ends, max_time, n_steps):
    assert isinstance(starts, torch.Tensor)
    assert isinstance(ends, torch.Tensor)

    ends[ends < 0] = max_time
    fps = n_steps / max_time
    times = (torch.arange(0, n_steps, device=labels.device, dtype=torch.float32) + .5) / fps
    after_start = starts.unsqueeze(1) <= times.unsqueeze(0)
    before_end = ends.unsqueeze(1) >= times.unsqueeze(0)
    # Find when you are inside of a word
    in_word = (after_start * before_end)
    # Find which word you are inside of
    word_to_use = in_word.to(torch.float32).argmax(0)
    # Get the label for that word, or mask out the label if in no word
    final_labels = labels[word_to_use] * in_word.any(0).reshape(-1, 1, 1)
    return final_labels


def generate_subset(n, batch, seed=0):
    np.random.seed(seed)
    return np.random.permutation(n)[:batch]


def channel_blur(t, window=5, std_dev=1):
    tb, tc, th, tw = t.shape
    x = torch.linspace(-2, 2, window, device=t.device, dtype=torch.float32)
    k = torch.exp((-x ** 2 / (2 * std_dev ** 2)))
    k = k / k.sum()
    pad = window // 2
    t_pad = F.pad(t, [0, 0, 0, 0, pad, pad], mode="replicate")
    tpb, tpc, tph, tpw = t_pad.shape
    flattened_t = t_pad.permute(0, 2, 3, 1).reshape(tpb * tph * tpw, 1, -1)
    return F.conv1d(flattened_t, k.reshape(1, 1, window)).reshape(tpb, tph, tpw, tc).permute(0, 3, 1, 2)


def time_blur(t, window=5, std_dev=1):
    tb, tc, tt = t.shape
    with torch.no_grad():
        x = torch.linspace(-2, 2, window, device=t.device, dtype=torch.float32)
        k = torch.exp((-x ** 2 / (2 * std_dev ** 2)))
        k = k / k.sum()
        k = k.reshape(1, 1, window).detach()
    pad = window // 2
    t_pad = F.pad(t, [pad, pad], mode="replicate")
    return F.conv1d(t_pad.reshape(tb * tc, 1, -1), k).reshape(tb, tc, tt)


def create_model_from_cfg(clazz, cfg, extra_args):
    import inspect
    expected_args = inspect.getfullargspec(clazz.__init__).args[1:]
    new_args = {k: v for k, v in {**cfg, **extra_args}.items() if k in expected_args}
    return clazz(**new_args)


def load_trained_model(chkpt_dir, extra_args, strict=True):
    from train_av_alignment import LitAVAligner
    model = LitAVAligner.load_from_checkpoint(chkpt_dir, **extra_args, strict=strict).cuda()
    return model


def flatten(l):
    return [item for sublist in l for item in sublist]


def flatten_preds(preds):
    results = {}
    for k in preds[0].keys():
        if k == "caption_labels":
            continue
        if isinstance(preds[0][k], torch.Tensor):
            results[k] = torch.cat([p[k] for p in preds], dim=0)
    if "caption" in preds[0]:
        results["caption"] = flatten([p["caption"] for p in preds])

    if "metadata" in preds[0]:
        results["frame_files"] = flatten([list(p["metadata"]["frame_files"][0]) for p in preds])
        results["audio_file"] = flatten([list(p["metadata"]["audio_file"]) for p in preds])
        results["id"] = flatten([list(p["metadata"]["id"]) for p in preds])
        results["index"] = torch.tensor(flatten([list(p["metadata"]["index"]) for p in preds]))

    return results


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

    @staticmethod
    def jvp(ctx: Any, *grad_inputs: Any) -> Any:
        pass

    @staticmethod
    def forward(ctx, inputs):
        ctx.save_for_backward(inputs)
        output = [torch.zeros_like(inputs) for _ in range(dist.get_world_size())]
        dist.all_gather(output, inputs)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (inputs,) = ctx.saved_tensors
        grad_out = torch.zeros_like(inputs)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out


class RollingAvg:

    def __init__(self, length, nonzero=False):
        self.length = length
        self.nonzero = nonzero
        self.metrics = defaultdict(lambda: deque(maxlen=self.length))

    def add(self, name, metric):
        if self.nonzero and metric == 0:
            return
        if isinstance(metric, torch.Tensor):
            metric = metric.detach()

        self.metrics[name].append(metric)

    def get(self, name):
        with torch.no_grad():
            return torch.tensor(list(self.metrics[name])).mean()

    def get_all(self):
        return {k: self.get(k) for k in self.metrics.keys()}

    def add_all(self, values):
        for k, v in values.items():
            self.add(k, v)

    def logall(self, log_func):
        for k in self.metrics.keys():
            log_func(k, self.get(k))


def gaussian_kernel(k, sigma):
    kernel = torch.tensor([math.exp(-0.5 * (x - (k // 2)) ** 2 / sigma ** 2) for x in range(k)], dtype=torch.float32)
    kernel /= kernel.sum()  # Normalize the kernel
    return kernel


def blur_dim(t, window=5, std_dev=1, dim=-1):
    shape = t.shape
    n_dims = len(shape)

    # Create the Gaussian kernel
    with torch.no_grad():
        x = torch.linspace(-2, 2, window, device=t.device, dtype=torch.float32)
        k = torch.exp(-x ** 2 / (2 * std_dev ** 2))
        k = k / k.sum()
        k = k.view(1, 1, window).detach()

    # Calculate padding
    pad = window // 2

    # Move the target dimension to the end
    permute_order = list(range(n_dims))
    permute_order.append(permute_order.pop(dim))
    t_permuted = t.permute(permute_order)

    # Flatten all dimensions except the last one
    new_shape = (-1, t_permuted.size(-1))
    t_flattened = t_permuted.reshape(new_shape)

    # Pad the tensor
    t_padded = F.pad(t_flattened.unsqueeze(1), (pad, pad), mode="replicate")

    # Apply convolution
    blurred = F.conv1d(t_padded, k)

    # Reshape back to original
    blurred = blurred.squeeze(1).reshape(*t_permuted.shape)
    blurred = blurred.permute([permute_order.index(i) for i in range(n_dims)])

    return blurred
