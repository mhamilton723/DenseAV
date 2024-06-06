import os
import re
from os.path import join

import torch



def get_latest(name, checkpoint_dir, extra_args=None):
    if extra_args is None:
        extra_args = dict()
    files = os.listdir(join(checkpoint_dir, name))
    steps = torch.tensor([int(f.split("step=")[-1].split(".")[0]) for f in files])
    selected = files[steps.argmax()]
    return dict(
        chkpt_name=os.path.join(name, selected),
        extra_args=extra_args)


DS_PARAM_REGEX = r'_forward_module\.(.+)'


def convert_deepspeed_checkpoint(deepspeed_ckpt_path: str, pl_ckpt_path: str = None):
    '''
    Creates a PyTorch Lightning checkpoint from the DeepSpeed checkpoint directory, while patching
    in parameters which are improperly loaded by the DeepSpeed conversion utility.
    deepspeed_ckpt_path: Path to the DeepSpeed checkpoint folder.
    pl_ckpt_path: Path to the reconstructed PyTorch Lightning checkpoint. If not specified, will be
        placed in the same directory as the DeepSpeed checkpoint directory with the same name but
        a .pt extension.
    Returns: path to the converted checkpoint.
    '''
    from pytorch_lightning.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict


    if not (deepspeed_ckpt_path.endswith('.ckpt') and os.path.isdir(deepspeed_ckpt_path)):
        raise ValueError(
            'args.ckpt_dir should point to the checkpoint directory'
            ' output by DeepSpeed (e.g. "last.ckpt" or "epoch=4-step=39150.ckpt").'
        )

    # Convert state dict to PyTorch format
    if not pl_ckpt_path:
        pl_ckpt_path = f'{deepspeed_ckpt_path[:-4]}pt'  # .ckpt --> .pt

    if not os.path.exists(pl_ckpt_path):
        convert_zero_checkpoint_to_fp32_state_dict(deepspeed_ckpt_path, pl_ckpt_path)

    # Patch in missing parameters that failed to be converted by DeepSpeed utility
    pl_ckpt = _merge_deepspeed_weights(deepspeed_ckpt_path, pl_ckpt_path)
    torch.save(pl_ckpt, pl_ckpt_path)

    return pl_ckpt_path


def get_optim_files(checkpoint_dir):
    files = sorted([f for f in os.listdir(checkpoint_dir) if "optim" in f])
    return [join(checkpoint_dir, f) for f in files]


def get_model_state_file(checkpoint_dir, zero_stage):
    f = [f for f in os.listdir(checkpoint_dir) if "model_states" in f][0]
    return join(checkpoint_dir, f)


def _merge_deepspeed_weights(deepspeed_ckpt_path: str, fp32_ckpt_path: str):
    '''
    Merges tensors with keys in the DeepSpeed checkpoint but not in the fp32_checkpoint
    into the fp32 state dict.
    deepspeed_ckpt_path: Path to the DeepSpeed checkpoint folder.
    fp32_ckpt_path: Path to the reconstructed
    '''
    from pytorch_lightning.utilities.deepspeed import ds_checkpoint_dir


    # This first part is based on pytorch_lightning.utilities.deepspeed.convert_zero_checkpoint_to_fp32_state_dict
    checkpoint_dir = ds_checkpoint_dir(deepspeed_ckpt_path)
    optim_files = get_optim_files(checkpoint_dir)
    optim_state = torch.load(optim_files[0], map_location='cpu')
    zero_stage = optim_state["optimizer_state_dict"]["zero_stage"]
    deepspeed_model_file = get_model_state_file(checkpoint_dir, zero_stage)

    # Start adding all parameters from DeepSpeed ckpt to generated PyTorch Lightning ckpt
    ds_ckpt = torch.load(deepspeed_model_file, map_location='cpu')
    ds_sd = ds_ckpt['module']

    fp32_ckpt = torch.load(fp32_ckpt_path, map_location='cpu')
    fp32_sd = fp32_ckpt['state_dict']

    for k, v in ds_sd.items():
        try:
            match = re.match(DS_PARAM_REGEX, k)
            param_name = match.group(1)
        except:
            print(f'Failed to extract parameter from DeepSpeed key {k}')
            continue

        v = v.to(torch.float32)
        if param_name not in fp32_sd:
            print(f'Adding parameter {param_name} from DeepSpeed state_dict to fp32_sd')
            fp32_sd[param_name] = v
        else:
            assert torch.allclose(v, fp32_sd[param_name].to(torch.float32), atol=1e-2)

    return fp32_ckpt


def get_version_and_step(f, i):
    step = f.split("step=")[-1].split(".")[0]
    if "-v" in step:
        [step, version] = step.split("-v")
    else:
        step, version = step, 0

    return int(version), int(step), i


def get_latest_ds(name, extra_args=None):
    if extra_args is None:
        extra_args = dict()
    files = os.listdir(f"../checkpoints/{name}")
    latest = sorted([get_version_and_step(f, i) for i, f in enumerate(files)], reverse=True)[0]
    selected = files[latest[-1]]
    # print(f"Selecting file: {selected}")
    ds_chkpt = join(name, selected)
    reg_chkpt = join(name + "_fp32", selected)
    reg_chkpt_path = join("../checkpoints", reg_chkpt)
    if not os.path.exists(reg_chkpt_path):
        os.makedirs(os.path.dirname(reg_chkpt_path), exist_ok=True)
        print(f"Checkpoint {reg_chkpt} does not exist, converting from deepspeed")
        convert_deepspeed_checkpoint(join("../checkpoints", ds_chkpt), reg_chkpt_path)
    return dict(
        chkpt_name=reg_chkpt,
        extra_args=extra_args)


def get_all_models_in_dir(name, checkpoint_dir, extra_args=None):
    ret = {}
    for model_dir in os.listdir(join(checkpoint_dir, name)):
        full_name = f"{name}/{model_dir}/train"
        # print(f'"{full_name}",')
        ret[full_name] = get_latest(full_name, checkpoint_dir, extra_args)
    return ret


def saved_model_dict(checkpoint_dir):
    model_info = {

        **get_all_models_in_dir(
            "9-5-23-mixed",
            checkpoint_dir,
            extra_args=dict(
                mixup_weight=0.0,
                sim_use_cls=False,
                audio_pool_width=1,
                memory_buffer_size=0,
                loss_leak=0.0)
        ),

        **get_all_models_in_dir(
            "1-23-24-rebuttal-heads",
            checkpoint_dir,
            extra_args=dict(
                loss_leak=0.0)
        ),

        **get_all_models_in_dir(
            "11-8-23",
            checkpoint_dir,
            extra_args=dict(loss_leak=0.0)),

        **get_all_models_in_dir(
            "10-30-23-3",
            checkpoint_dir,
            extra_args=dict(loss_leak=0.0)),

        "davenet": dict(
            chkpt_name=None,
            extra_args=dict(
                audio_blur=1,
                image_model_type="davenet",
                image_aligner_type=None,
                audio_model_type="davenet",
                audio_aligner_type=None,
                audio_input="davenet_spec",
                use_cached_embs=False,
                dropout=False,
                sim_agg_heads=1,
                nonneg_sim=False,
                audio_lora=False,
                image_lora=False,
                norm_vectors=False,
            ),
            data_args=dict(
                use_cached_embs=False,
                use_davenet_spec=True,
                override_target_length=20,
                audio_model_type="davenet",
            ),
        ),

        "cavmae": dict(
            chkpt_name=None,
            extra_args=dict(
                audio_blur=1,
                image_model_type="cavmae",
                image_aligner_type=None,
                audio_model_type="cavmae",
                audio_aligner_type=None,
                audio_input="spec",
                use_cached_embs=False,
                sim_agg_heads=1,
                dropout=False,
                nonneg_sim=False,
                audio_lora=False,
                image_lora=False,
                norm_vectors=False,
                learn_audio_cls=False,
                sim_agg_type="cavmae",
            ),
            data_args=dict(
                use_cached_embs=False,
                use_davenet_spec=True,
                audio_model_type="cavmae",
                override_target_length=10,
            ),
        ),

        "imagebind": dict(
            chkpt_name=None,
            extra_args=dict(
                audio_blur=1,
                image_model_type="imagebind",
                image_aligner_type=None,
                audio_model_type="imagebind",
                audio_aligner_type=None,
                audio_input="spec",
                use_cached_embs=False,
                sim_agg_heads=1,
                dropout=False,
                nonneg_sim=False,
                audio_lora=False,
                image_lora=False,
                norm_vectors=False,
                learn_audio_cls=False,
                sim_agg_type="imagebind",
            ),
            data_args=dict(
                use_cached_embs=False,
                use_davenet_spec=True,
                audio_model_type="imagebind",
                override_target_length=10,
            ),
        ),

    }

    model_info["denseav_language"] = model_info["10-30-23-3/places_base/train"]
    model_info["denseav_sound"] = model_info["11-8-23/hubert_1h_asf_cls_full_image_train_small_lr/train"]
    model_info["denseav_2head"] = model_info["1-23-24-rebuttal-heads/mixed-2h/train"]

    return model_info
