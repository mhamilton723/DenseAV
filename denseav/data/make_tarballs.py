import glob
import os
import tarfile
from glob import glob
from io import BytesIO
from os.path import join

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pathlib import Path

from denseav.shared import batch

import tempfile
import shutil


class Tarballer(Dataset):

    def __init__(self, source, target, n):
        source_path = Path(source)
        self.frames = [f.relative_to(source_path) for f in source_path.rglob('*') if f.is_file()]
        assert (len(self.frames) > 0)
        self.source = source
        self.target_dir = target
        self.batched = list(batch(self.frames, n))
        os.makedirs(self.target_dir, exist_ok=True)

    def __len__(self):
        return len(self.batched)

    def __getitem__(self, item):
        with tarfile.open(join(self.target_dir, f"{item}.tar"), "w") as tar:
            for relpath in self.batched[item]:
                abs_path = os.path.join(self.source, str(relpath))  # Convert to string here
                with open(abs_path, "rb") as file:
                    file_content = file.read()
                info = tarfile.TarInfo(name=str(relpath))  # Convert to string here
                info.size = len(file_content)
                tar.addfile(info, fileobj=BytesIO(file_content))

        return 0


class UnTarballer:

    def __init__(self, archive_dir, target_dir, remove_source=False):
        self.tarballs = sorted(glob(join(archive_dir, "*.tar")))
        self.target_dir = target_dir
        self.remove_source = remove_source  # New flag to determine if source tarball should be removed
        os.makedirs(self.target_dir, exist_ok=True)

    def __len__(self):
        return len(self.tarballs)

    def __getitem__(self, item):
        with tarfile.open(self.tarballs[item], "r") as tar:
            # Create a unique temporary directory inside the target directory
            with tempfile.TemporaryDirectory(dir=self.target_dir) as tmpdirname:
                tar.extractall(tmpdirname)  # Extract to the temporary directory

                # Move contents from temporary directory to final target directory
                for src_dir, dirs, files in os.walk(tmpdirname):
                    dst_dir = src_dir.replace(tmpdirname, self.target_dir, 1)
                    os.makedirs(dst_dir, exist_ok=True)
                    for file_ in files:
                        src_file = os.path.join(src_dir, file_)
                        dst_file = os.path.join(dst_dir, file_)
                        shutil.move(src_file, dst_file)

        # Remove the source tarball if the flag is set to True
        if self.remove_source:
            os.remove(self.tarballs[item])

        return 0

def untar_all(archive_dir, target_dir, remove_source):
    loader = DataLoader(UnTarballer(archive_dir, target_dir, remove_source), num_workers=24)
    for _ in tqdm(loader):
        pass


if __name__ == "__main__":
    # loader = DataLoader(Tarballer(
    #     join("/pytorch-data", "audioset-raw", "audio"),
    #     join("/pytorch-data", "audioset-raw", "audio_archives")
    # ), num_workers=24)

    # loader = DataLoader(Tarballer(
    #     join("/pytorch-data", "audioset-raw", "frames"),
    #     join("/pytorch-data", "audioset-raw", "frame_archives"),
    #     5000
    # ), num_workers=24)

    # loader = DataLoader(Tarballer(
    #     join("/pytorch-data", "ADE20KLabels"),
    #     join("/pytorch-data", "ADE20KLabelsAr"),
    #     100
    # ), num_workers=24)
    #
    # for _ in tqdm(loader):
    #     pass
    #
    # #
    #
    untar_all(
        join("/pytorch-data", "audioset-raw", "frame_archives"),
        join("/pytorch-data", "audioset-raw", "frames_4"))
