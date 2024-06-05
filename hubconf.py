# hubconf.py
from denseav.train_av_alignment import LitAVAligner

dependencies = ['torch', 'torchvision', 'PIL']  # List any dependencies here


def _load_base(model_name):
    model = LitAVAligner.load_from_checkpoint(
        f"https://marhamilresearch4.blob.core.windows.net/denseav-public/hub/{model_name}.ckpt",
        **{'loss_leak': 0.0, 'use_cached_embs': False},
        strict=True)
    model.set_full_train(True)
    return model


def denseav_2head():
    return _load_base("denseav_2head")


def denseav_language():
    return _load_base("denseav_language")


def denseav_sound():
    return _load_base("denseav_sound")
