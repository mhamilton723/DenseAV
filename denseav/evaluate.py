from os.path import join
import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from denseav.data.AVDatasets import AVDataModule
from denseav.shared import load_trained_model


@hydra.main(config_path="configs", config_name="av_align.yaml")
def my_app(cfg: DictConfig) -> None:
    from saved_models import saved_model_dict

    seed_everything(0)
    print(OmegaConf.to_yaml(cfg))

    models_to_eval = [
        "denseav_language",
        "denseav_sound",
    ]

    checkpoint_dir = "../checkpoints"
    saved_models = saved_model_dict(checkpoint_dir)
    for model_name in models_to_eval:
        model_info = saved_models[model_name]
        extra_data_args = model_info["data_args"] if "data_args" in model_info else {}
        model_info["extra_args"]["output_root"] = "../"
        model_info["extra_args"]["neg_audio"] = False
        model_info["extra_args"]["image_mixup"] = 0.0

        model = load_trained_model(join(checkpoint_dir, model_info["chkpt_name"]), model_info["extra_args"])
        model.set_full_train(True)

        if model.image_model_type == "dinov2":
            load_size = cfg.load_size * 2
        else:
            load_size = cfg.load_size

        if model.image_model_type == "davenet":
            batch_size = cfg.batch_size // 2
        elif model.image_model_type == "imagebind":
            batch_size = cfg.batch_size
        else:
            batch_size = cfg.batch_size

        print(load_size)

        data_args = dict(
            dataset_name=cfg.dataset_name,
            load_size=load_size,
            image_aug=cfg.image_aug,
            audio_aug=cfg.audio_aug,
            audio_model_type=model.audio_model_type,
            pytorch_data_dir=cfg.pytorch_data_dir,
            use_cached_embs=model.use_cached_embs,
            batch_size=batch_size,
            num_workers=cfg.num_workers,
            extra_audio_masking=False,
            use_original_val_set=False,
            use_extra_val_sets=True,
            use_caption=True,
            data_for_plotting=False,
            n_frames=None,
            audio_level=False,
            neg_audio=False,
            quad_mixup=0.0,
            bg_mixup=0.0,
            patch_mixup=0.0,
            patch_size=8,
        )
        data_args = {**data_args, **extra_data_args}

        datamodule = AVDataModule(**data_args)
        log_dir = join(cfg.output_root, "logs", "evaluate", model_name)
        print(log_dir)
        tb_logger = TensorBoardLogger(log_dir, default_hp_metric=False)
        trainer = Trainer(
            accelerator='gpu',
            strategy="ddp",
            devices=cfg.num_gpus,
            logger=tb_logger)
        trainer.validate(model, datamodule)


if __name__ == "__main__":
    my_app()
