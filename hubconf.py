# hubconf.py
from denseav.train import LitAVAligner

dependencies = ['torch', 'torchvision', 'PIL', 'denseav']  # List any dependencies here


def _load_base(model_name):
    model = LitAVAligner.load_from_checkpoint(
        f"https://marhamilresearch4.blob.core.windows.net/denseav-public/hub/{model_name}.ckpt",
        **{'loss_leak': 0.0, 'use_cached_embs': False},
        strict=True)
    model.set_full_train(True)
    return model


def sound_and_language():
    return _load_base("denseav_2head")


def language():
    return _load_base("denseav_language")


def sound():
    return _load_base("denseav_sound")
