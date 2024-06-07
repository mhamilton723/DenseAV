import os
from collections import deque
from itertools import combinations
from os.path import join

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributed as dist
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from peft import get_peft_model, LoraConfig
from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities import grad_norm
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, SequentialLR, LambdaLR
from torchmetrics.functional.classification import binary_average_precision

from huggingface_hub import PyTorchModelHubMixin

from denseav.aggregators import get_aggregator
from denseav.aligners import get_aligner, ProgressiveGrowing
from denseav.constants import *
from denseav.data.AVDatasets import AVDataModule
from denseav.shared import flatten_preds, GatherLayer, \
    get_image_featurizer, get_audio_featurizer, RollingAvg, create_model_from_cfg

torch.multiprocessing.set_sharing_strategy('file_system')


def _imposter_indices_helper(true_indices: torch.Tensor, samples: torch.Tensor):
    mask = (true_indices == samples).to(torch.int64)
    n = mask.shape[0]

    if not mask.any():
        return samples
    else:
        new_samples = torch.randint(0, n, size=(n,), device=true_indices.device)
        comb_samples = mask * new_samples + (1 - mask) * samples
        return _imposter_indices_helper(true_indices, comb_samples)


def imposter_indices(n, device):
    return _imposter_indices_helper(
        torch.arange(0, n, device=device),
        torch.randint(0, n, size=(n,), device=device))


def get_sim_per_row(image_outputs, audio_outputs, n_frames, sim_type):
    max_t = audio_outputs.shape[-1]
    oh = F.one_hot(n_frames - 1, num_classes=max_t)
    audio_mask = 1 - torch.cumsum(oh, dim=1)
    audio_mask = F.pad(audio_mask, [1, 0], value=1)[:, :max_t].to(audio_outputs.dtype)

    full_sim = torch.einsum("bct,bchw->bthw", audio_outputs, image_outputs)
    expanded_am = audio_mask.unsqueeze(-1).unsqueeze(-1)

    if sim_type.endswith("mi"):
        offset = 10 * (full_sim.max() - full_sim.min())
        full_sim = (full_sim - ((1 - expanded_am) * offset)).max(1, keepdim=True).values

    if sim_type.startswith("mi"):
        full_sim = full_sim.max(-1, keepdim=True).values.max(-2, keepdim=True).values

    if sim_type.endswith("sa"):
        full_sim = (full_sim * (expanded_am / expanded_am.sum(1, keepdim=True).clamp_min(1))).sum(1, keepdim=True)

    return full_sim.mean(dim=[1, 2, 3])


def sampled_margin_rank_loss(image_outputs, audio_outputs, n_frames, sim_type, margin=1.):
    """
    Computes the triplet margin ranking loss for each anchor image/caption pair
    The impostor image/caption is randomly sampled from the minibatch
    """
    assert (image_outputs.dim() == 4)
    assert (audio_outputs.dim() == 3)
    n = image_outputs.size(0)
    imp_ind_i = imposter_indices(n, image_outputs.device)
    imp_ind_a = imposter_indices(n, image_outputs.device)
    true_sim = get_sim_per_row(image_outputs, audio_outputs, n_frames, sim_type)
    imp_sim_i = get_sim_per_row(image_outputs[imp_ind_i], audio_outputs, n_frames, sim_type)
    imp_sim_a = get_sim_per_row(image_outputs, audio_outputs[imp_ind_a], n_frames[imp_ind_a], sim_type)
    a2i_loss = (margin + imp_sim_i - true_sim).clamp_min(0)
    i2a_loss = (margin + imp_sim_a - true_sim).clamp_min(0)
    return (a2i_loss + i2a_loss).mean() / 2


class SimilarityCalibrator(torch.nn.Module):

    def __init__(self, cal_init, max_w=100, min_w=.01, subtract_mean=True, use_bias=False):
        super().__init__()
        self.max_w = max_w
        self.min_w = min_w
        self.w = torch.nn.Parameter(torch.tensor([cal_init]).log())

        self.use_bias = use_bias
        if self.use_bias:
            self.b = torch.nn.Parameter(torch.tensor([0.0]))

        self.subtract_mean = subtract_mean

    def get_w(self):
        return torch.exp(self.w).clamp_max(self.max_w).clamp_min(self.min_w)

    def forward(self, x):
        sims = self.get_w() * x

        if self.use_bias:
            sims = sims + self.b

        if self.subtract_mean:
            return sims - sims.mean()
        else:
            return sims


class SpatialDropout(torch.nn.Module):

    def __init__(self, p, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p = p

    def forward(self, x):
        b, c, h, w = x.shape
        dropout = torch.rand((b, 1, h, w), dtype=x.dtype, device=x.device) > self.p

        if self.training:
            return x * dropout
        else:
            return x


class LitAVAligner(PyTorchModelHubMixin, pl.LightningModule):
    def __init__(self,
                 code_dim,
                 image_model_type,
                 image_model_token_type,
                 image_aligner_type,
                 image_pool_width,
                 audio_model_type,
                 audio_aligner_type,
                 audio_pool_width,
                 audio_lora,
                 audio_lora_rank,
                 image_lora,
                 image_lora_rank,
                 gradient_clipping,
                 learn_audio_cls,
                 silence_l1,
                 silence_l2,
                 tv_weight,
                 nonneg_sim,
                 nonneg_pressure,
                 pretrain_lr,
                 lr,
                 lr_warmup,
                 lr_schedule,
                 lr_cycle_length,
                 optimizer,
                 gather_tensors,
                 sim_agg_type,
                 sim_agg_heads,
                 sim_use_cls,
                 disentangle_weight,
                 norm_vectors,
                 cal_init,
                 cal_balance_weight,
                 loss_type,
                 loss_margin,
                 mask_silence,
                 finetune_image_model,
                 finetune_audio_model,
                 use_cached_embs,
                 output_root,
                 neg_audio,
                 neg_audio_weight,
                 head_agg,
                 adaptive_clipping,
                 specialization_weight,
                 spatial_dropout,
                 channel_dropout,
                 mixup_weight,
                 memory_buffer_size,
                 loss_leak,
                 ):
        super().__init__()

        self.code_dim = code_dim
        self.image_model_type = image_model_type
        self.image_model_token_type = image_model_token_type
        self.image_aligner_type = image_aligner_type
        self.image_pool_width = image_pool_width
        self.audio_model_type = audio_model_type
        self.audio_aligner_type = audio_aligner_type
        self.audio_pool_width = audio_pool_width

        self.gradient_clipping = gradient_clipping
        self.learn_audio_cls = learn_audio_cls
        self.silence_l1 = silence_l1
        self.silence_l2 = silence_l2

        self.tv_weight = tv_weight
        self.nonneg_sim = nonneg_sim
        self.nonneg_pressure = nonneg_pressure
        self.pretrain_lr = pretrain_lr
        self.lr = lr
        self.lr_warmup = lr_warmup
        self.lr_schedule = lr_schedule
        self.lr_cycle_length = lr_cycle_length
        self.optimizer = optimizer
        self.gather_tensors = gather_tensors
        self.sim_agg_type = sim_agg_type
        self.sim_agg_heads = sim_agg_heads
        self.sim_use_cls = sim_use_cls
        self.disentangle_weight = disentangle_weight

        self.norm_vectors = norm_vectors
        self.cal_init = cal_init
        self.cal_balance_weight = cal_balance_weight
        self.loss_type = loss_type
        self.loss_margin = loss_margin
        self.mask_silence = mask_silence
        self.finetune_image_model = finetune_image_model
        self.finetune_audio_model = finetune_audio_model
        self.use_cached_embs = use_cached_embs
        self.output_root = output_root
        self.audio_lora = audio_lora
        self.audio_lora_rank = audio_lora_rank
        self.image_lora = image_lora
        self.image_lora_rank = image_lora_rank
        self.neg_audio = neg_audio
        self.neg_audio_weight = neg_audio_weight
        self.head_agg = head_agg

        self.adaptive_clipping = adaptive_clipping
        self.specialization_weight = specialization_weight
        self.spatial_dropout = spatial_dropout
        self.channel_dropout = channel_dropout
        self.mixup_weight = mixup_weight

        self.memory_buffer_size = memory_buffer_size
        self.memory_buffer = deque(maxlen=self.memory_buffer_size)
        self.loss_leak = loss_leak

        if self.audio_model_type in {"audiomae", "audiomae-finetuned", "cavmae", "cavmae-mixed", "imagebind"}:
            self.audio_input = "spec"
        elif self.audio_model_type == "davenet":
            self.audio_input = "davenet_spec"
        elif self.audio_model_type == "fnac":
            self.audio_input = "fnac_spec"
        else:
            self.audio_input = "audio"

        extra_model_args = dict(output_root=output_root)

        self.image_model, _, self.image_feat_dim = get_image_featurizer(
            image_model_type, token_type=self.image_model_token_type, **extra_model_args)

        self.image_model.eval()
        if not self.finetune_image_model:
            for param in self.image_model.parameters():
                param.requires_grad = False

        if image_model_type in {"cavmae", "cavmae-mixed", "imagebind", "fnac"}:
            extra_model_args["model"] = self.image_model.model

        if use_cached_embs:
            _, self.audio_feat_dim = get_audio_featurizer(audio_model_type, **extra_model_args)
        else:
            self.audio_model, self.audio_feat_dim = get_audio_featurizer(audio_model_type, **extra_model_args)

            self.audio_model.eval()
            if not self.finetune_audio_model:
                for param in self.audio_model.parameters():
                    param.requires_grad = False

        if self.image_lora:
            if self.image_model_type in {"sam", "dino8", "dinov2", "cavmae", "cavmae-mixed"}:
                target_modules = ["qkv"]
            elif self.image_model_type == "clip":
                target_modules = ["out_proj"]
            elif self.image_model_type == "imagebind":
                target_modules = ["out_proj", "fc1", "fc2"]
            else:
                target_modules = ["q", "k", "v"]

            peft_config = LoraConfig(
                target_modules=target_modules,
                inference_mode=False,
                r=image_lora_rank,
                lora_alpha=32,
                lora_dropout=0.1
            )
            self.image_model = get_peft_model(self.image_model, peft_config)
            self.image_model.print_trainable_parameters()

        if self.audio_lora:
            if self.audio_model_type == "hubert":
                target_modules = ["q_proj", "k_proj", "v_proj"]
            else:
                target_modules = ["q", "k", "v"]

            peft_config = LoraConfig(
                inference_mode=False,
                target_modules=target_modules,
                r=audio_lora_rank,
                lora_alpha=32,
                lora_dropout=0.1
            )
            self.audio_model = get_peft_model(self.audio_model, peft_config)
            self.audio_model.print_trainable_parameters()

        shared_aligner_args = dict(out_dim=self.code_dim)

        self.audio_aligner = get_aligner(
            self.audio_aligner_type, self.audio_feat_dim, **shared_aligner_args)
        self.image_aligner = get_aligner(
            self.image_aligner_type, self.image_feat_dim, **shared_aligner_args)

        if self.loss_type == "nce":
            self.sim_cal = SimilarityCalibrator(self.cal_init, subtract_mean=True, use_bias=False)
        else:
            self.sim_cal = SimilarityCalibrator(self.cal_init, subtract_mean=False, use_bias=True)

        if self.learn_audio_cls:
            self.audio_cls = torch.nn.Parameter(torch.randn(self.audio_feat_dim))

        if self.spatial_dropout > 0.0:
            self.spatial_dropout_layer = SpatialDropout(self.spatial_dropout)

        if self.channel_dropout > 0.0:
            self.channel_dropout_layer = torch.nn.Dropout2d(self.channel_dropout)

        self.sim_agg = get_aggregator(
            self.sim_agg_type,
            self.nonneg_sim,
            self.mask_silence,
            self.sim_agg_heads,
            self.head_agg,
            self.sim_use_cls,
            dim=self.image_feat_dim
        )

        self.hparams_logged = False
        self.rolling_avg = RollingAvg(50)
        self.grad_avg = RollingAvg(50, nonzero=True)

        self.save_hyperparameters()

    def set_full_train(self, full_train):
        self.full_train = full_train

    def prep_feats(self, feats, is_audio):

        if not is_audio and self.training and self.image_pool_width > 1:
            feats = torch.nn.AvgPool2d(self.image_pool_width)(feats)

        if is_audio and self.training and self.audio_pool_width > 1:
            feats = torch.nn.AvgPool2d((1, self.audio_pool_width))(feats)

        if self.norm_vectors:
            feats = F.normalize(feats, dim=1)

        return feats

    def on_before_optimizer_step(self, optimizer, optimizer_idx):
        norms = grad_norm(self, norm_type=2)
        avg_grads = self.grad_avg.get_all()
        params = {
            f"grad_2.0_norm/{name}": p
            for name, p in self.named_parameters()
            if p.grad is not None
        }

        if self.adaptive_clipping:
            for k in norms.keys():
                if k in params:
                    avg_grad = max(avg_grads.get(k, norms[k]), 1e-5)
                    if self.global_step > 10 and norms[k] > avg_grad * 5:
                        print(f"Bad grad for {k}: {norms[k]} scaling to {avg_grad * 5}")
                        torch.nn.utils.clip_grad_norm_(params[k], avg_grad * 5)
                        norms[k] = avg_grad * 5

                    if norms[k] > self.gradient_clipping:
                        # print(f"Bad grad for {k}: {norms[k]} scaling to {self.gradient_clipping}")
                        torch.nn.utils.clip_grad_norm_(params[k], self.gradient_clipping)

        # self.grad_avg.add_all(norms)
        # self.log_dict(norms)

    def interpolate_mask(self, mask, target_length, discrete):
        b, t = mask.shape

        mask = F.interpolate(mask.reshape(b, 1, 1, t), (1, target_length), mode="bilinear") \
            .reshape(b, target_length)

        if discrete:
            mask = mask > 0.01
            sums = mask.sum(1)
            all_zeros = torch.where(sums == 0)[0]
            if len(all_zeros) > 0:
                print("Fixing a bad mask")
                for entry in all_zeros:
                    mask[entry, torch.randint(0, target_length - 1, size=())] = True
        else:
            return mask
        return mask

    def forward_audio(self, batch):
        if self.use_cached_embs:
            audio_feats = batch["audio_emb"]
            if "audio_cls" in batch:
                audio_cls = batch["audio_cls"]
            else:
                audio_cls = None
        else:
            audio = batch[self.audio_input]

            if self.full_train:
                audio_feats, audio_cls = self.audio_model(audio, include_cls=True)
            else:
                with torch.no_grad():
                    audio_feats, audio_cls = self.audio_model(audio, include_cls=True)

        mask = batch[AUDIO_MASK] if AUDIO_MASK in batch else torch.ones_like(audio)
        pos_mask = batch[AUDIO_POS_MASK] if AUDIO_POS_MASK in batch else torch.ones_like(audio)

        if self.learn_audio_cls:
            assert audio_cls is None
            audio_cls = torch.broadcast_to(self.audio_cls.unsqueeze(0), (audio_feats.shape[0], audio_feats.shape[1]))

        aligned_audio_feats, aligned_audio_cls = self.audio_aligner(audio_feats, audio_cls)

        if self.channel_dropout > 0.0:
            aligned_audio_feats = self.channel_dropout_layer(aligned_audio_feats)

        aligned_audio_feats = self.prep_feats(aligned_audio_feats, is_audio=True)
        audio_mask = self.interpolate_mask(mask, aligned_audio_feats.shape[-1], True)
        audio_pos_mask = self.interpolate_mask(pos_mask, aligned_audio_feats.shape[-1], False)

        ret = {
            AUDIO_MASK: audio_mask,
            AUDIO_POS_MASK: audio_pos_mask,
            AUDIO_FEATS: aligned_audio_feats,
        }

        if aligned_audio_cls is not None:
            ret[AUDIO_CLS] = aligned_audio_cls

        return ret

    # @autocast(device_type="cuda", enabled=False)
    def forward_image(self, batch, max_batch_size=None):

        with torch.no_grad():
            image = batch[IMAGE_INPUT]
            b, nf, c, h, w = image.shape
            image = image.reshape(b * nf, c, h, w)

            if max_batch_size is None:
                max_batch_size = image.shape[0]

            chunks = [image[i:i + max_batch_size] for i in range(0, image.shape[0], max_batch_size)]

            all_image_feats = []
            all_image_cls = []

            for chunk in chunks:
                if self.full_train:
                    image_feats, image_cls = self.image_model(chunk, include_cls=True)
                else:
                    with torch.no_grad():
                        image_feats, image_cls = self.image_model(chunk, include_cls=True)

                aligned_image_feats, aligned_image_cls = self.image_aligner(image_feats, image_cls)

                all_image_feats.append(aligned_image_feats)
                all_image_cls.append(aligned_image_cls)

            # Stitch the chunks back together
            aligned_image_feats = torch.cat(all_image_feats, dim=0)
            aligned_image_cls = torch.cat(all_image_cls, dim=0)

        if self.channel_dropout > 0.0:
            aligned_image_feats = self.channel_dropout_layer(aligned_image_feats)

        if self.spatial_dropout > 0.0:
            aligned_image_feats = self.spatial_dropout_layer(aligned_image_feats)

        aligned_image_feats = self.prep_feats(aligned_image_feats, is_audio=False)
        ret = {IMAGE_FEATS: aligned_image_feats}

        if IMAGE_MASK in batch:
            with torch.no_grad():
                mask = batch[IMAGE_MASK]
                mask = mask.reshape(b * nf, 1, h, w)
                b, c, h, w = aligned_image_feats.shape
                mask = F.adaptive_avg_pool2d(mask.to(aligned_image_feats), output_size=(h, w))
                ret[IMAGE_MASK] = mask

        if aligned_image_cls is not None:
            ret[IMAGE_CLS] = aligned_image_cls

        return ret

    def forward(self, batch):
        audio_feat_dict = self.forward_audio(batch)
        image_feat_dict = self.forward_image(batch)
        return {**image_feat_dict, **audio_feat_dict}

    def contrast_loss(self, sims):
        b = sims.shape[0]
        sims = sims - torch.eye(b, b, device=sims.device) * self.loss_margin
        sims_1 = sims
        sims_2 = sims.permute(1, 0)

        if self.loss_leak > 0.0:
            id = torch.eye(sims_1.shape[0], sims_1.shape[1], device=sims.device, dtype=sims.dtype)
            label_mask = id * (1 - self.loss_leak)
            label_mask += (1 - id) * self.loss_leak / (sims_1.shape[0] - 1)
            label_mask /= label_mask.sum(dim=1, keepdim=True)
        else:
            label_mask = torch.eye(sims_1.shape[0], sims_1.shape[1], device=sims.device, dtype=sims.dtype)

        labels = torch.arange(0, sims.shape[0], device=sims.device)
        self.rolling_avg.add(f"acc/1", (sims.argmax(dim=1) == labels).to(sims).mean())
        self.rolling_avg.add(f"acc/2", (sims.argmax(dim=0) == labels).to(sims).mean())

        if self.loss_type == "margin":
            margin_loss_tensor = (sims - torch.diag(sims)).clamp_min(0)
            margin_loss = margin_loss_tensor.mean()
            self.rolling_avg.add(f"loss/frac_nonzero", (margin_loss_tensor > 0).to(sims).mean())
            self.rolling_avg.add(f"loss/margin", margin_loss)
            return margin_loss
        elif self.loss_type == "ce":
            ce_loss = 1 / 2 * F.cross_entropy(sims_1, labels) + \
                      1 / 2 * F.cross_entropy(sims_2, labels)
            self.rolling_avg.add(f"loss/ce", ce_loss)
            return ce_loss
        elif self.loss_type == "bce":
            bce_loss = F.binary_cross_entropy_with_logits(sims_1.flatten(), label_mask.flatten())
            self.rolling_avg.add(f"loss/bce", bce_loss)
            return bce_loss
        elif self.loss_type == "nce":
            nce_loss = 1 / 2 * (-F.log_softmax(sims_1, dim=-1) * label_mask).sum(1).mean() + \
                       1 / 2 * (-F.log_softmax(sims_2, dim=-1) * label_mask).sum(1).mean()
            self.rolling_avg.add(f"loss/nce", nce_loss)
            return nce_loss
        else:
            raise ValueError(f"Unknown loss type {self.loss_type}")

    def loss(self, preds):
        image_feats = preds[IMAGE_FEATS]
        audio_feats = preds[AUDIO_FEATS]
        audio_mask = preds[AUDIO_MASK]
        image_mask = preds[IMAGE_MASK]
        audio_pos_mask = preds[AUDIO_POS_MASK]
        if DATA_SOURCE in preds:
            source = preds[DATA_SOURCE].to(torch.int64)
        else:
            source = None

        uncal_sims = self.sim_agg(preds, agg_heads=True)
        sims = self.sim_cal(uncal_sims)

        _mask = 1 - torch.eye(sims.shape[0], device=sims.device)
        self.log(f"sim/pos", torch.diag(sims).mean())
        self.log(f"sim/neg", (sims * _mask).sum() / (_mask.sum()))
        self.log(f"sim/uncal_pos", torch.diag(uncal_sims).mean())
        self.log(f"sim/uncal_neg", (uncal_sims * _mask).sum() / (_mask.sum()))

        b, c, h, w = image_feats.shape
        b, c, f, t = audio_feats.shape
        n_samples = 250

        nh = self.sim_agg_heads
        image_feats_by_head = image_feats.reshape(b, self.sim_agg_heads, c // nh, h, w)
        audio_feats_by_head = audio_feats.reshape(b, self.sim_agg_heads, c // nh, f, t)

        def maybe_clamp(t):
            return t.clamp_min(0) if self.nonneg_sim else t

        paired_sim_raw = self.sim_agg.get_pairwise_sims(preds, raw=True, agg_sim=False, agg_heads=False)
        paired_sim = maybe_clamp(paired_sim_raw)

        loss = 0.0

        if self.nonneg_pressure:
            afb, afk, afc, aff, aft = audio_feats_by_head.shape
            ifb, ifk, ifc, ifh, ifw = image_feats_by_head.shape
            assert (afb == ifb)

            device = audio_feats_by_head.device
            random_b = torch.randint(0, afb, size=(n_samples,), device=device)
            random_t = torch.randint(0, aft, size=(n_samples,), device=device)
            random_f = torch.randint(0, aff, size=(n_samples,), device=device)
            random_h = torch.randint(0, ifh, size=(n_samples,), device=device)
            random_w = torch.randint(0, ifw, size=(n_samples,), device=device)

            random_audio_feats = audio_feats_by_head[random_b, :, :, random_f, random_t]
            random_image_feats = image_feats_by_head[random_b, :, :, random_h, random_w]
            random_sim_raw = torch.einsum("bkc,dkc->bdk", random_audio_feats, random_image_feats)

            nonneg_loss = random_sim_raw.clamp_max(0).square().mean()
            self.rolling_avg.add(f"loss/nonneg", nonneg_loss)
            loss += nonneg_loss * self.nonneg_pressure

        if self.silence_l1 > 0 or self.silence_l2 > 0:
            masked_b, masked_t = torch.where(~audio_mask)
            if len(masked_b) > n_samples:
                subset = torch.randperm(len(masked_b))[:n_samples]
                masked_b = masked_b[subset]
                masked_t = masked_t[subset]

            if len(masked_b) == n_samples:
                silent_audio_feats = audio_feats_by_head[masked_b, :, :, :, masked_t].mean(-1)  # d k c
                silence_tensor = maybe_clamp(
                    torch.einsum("bkchw,dkc->bkdhw", image_feats_by_head, silent_audio_feats))

                silence_l1_loss = silence_tensor.abs().mean()
                self.rolling_avg.add(f"loss/silence_l1", silence_l1_loss)
                loss += silence_l1_loss * self.silence_l1

                silence_l2_loss = silence_tensor.square().mean()
                self.rolling_avg.add(f"loss/silence_l2", silence_l2_loss)
                loss += silence_l2_loss * self.silence_l2
            else:
                pass

        if self.neg_audio_weight > 0 and self.neg_audio:
            b, t = audio_pos_mask.shape
            negative_weight = ((1 - audio_pos_mask) * audio_mask.to(sims)).reshape(b, 1, 1, 1, 1, t)
            negative_weight = torch.broadcast_to(negative_weight, paired_sim.shape)
            if negative_weight.sum() > 0:
                neg_audio_loss = (paired_sim.square() * negative_weight).sum() \
                                 / negative_weight.sum().clamp_min(0.1)
                self.rolling_avg.add(f"loss/neg_audio", neg_audio_loss)
                self.rolling_avg.add(f"loss/neg_weight_avg", negative_weight.mean())
                loss += neg_audio_loss * self.neg_audio_weight
            else:
                print("WARNING: No negative samples found in batch")

        if self.tv_weight > 0:
            tv_loss = (paired_sim[:, :, :, :, :, 1:] - paired_sim[:, :, :, :, :, :-1]).square().mean()
            self.rolling_avg.add(f"loss/tv", tv_loss)
            loss += tv_loss * self.tv_weight

        self.log(f"cal/w", self.sim_cal.get_w())
        if self.cal_balance_weight > 0.0:
            cal_balance = (np.log(self.cal_init) - torch.log(self.sim_cal.get_w().clamp_min(.00000001))) \
                .clamp_min(0).square().mean()
            self.rolling_avg.add(f"loss/cal_balance", cal_balance)
            loss += cal_balance * self.cal_balance_weight

        if self.disentangle_weight > 0.0:
            assert source is not None
            assert self.sim_agg_heads % 2 == 0

            dilation = self.sim_agg_heads // 2
            sources_oh = F.one_hot(source, num_classes=2)
            b, h = sources_oh.shape
            sources_mask = 1 - torch.broadcast_to(sources_oh.unsqueeze(-1), (b, h, dilation)) \
                .reshape(b, h * dilation).to(paired_sim)
            disentangle_loss = torch.einsum("bkhwft,bk->bhwft", paired_sim, sources_mask).square().mean()
            self.rolling_avg.add(f"loss/disentangle", disentangle_loss)
            loss += disentangle_loss * self.disentangle_weight

        if self.specialization_weight > 0.0 and self.sim_agg_heads > 1:
            total_specialization_loss = 0.0
            combos = list(combinations(range(self.sim_agg_heads), 2))
            for i, j in combos:
                specialization_loss_pair = (paired_sim[:, i].abs() * paired_sim[:, j].abs()).mean()
                total_specialization_loss += specialization_loss_pair
            avg_specialization_loss = total_specialization_loss / len(combos)
            self.rolling_avg.add(f"loss/specialize", avg_specialization_loss)
            loss += avg_specialization_loss * self.specialization_weight

        if self.mixup_weight > 0.0:
            b, _, h, w = image_mask.shape
            neg_img_mask = torch.broadcast_to(
                1 - image_mask.to(paired_sim).reshape(b, 1, h, w, 1, 1),
                paired_sim.shape)
            image_mixup_loss = (paired_sim * neg_img_mask).square().sum() / neg_img_mask.sum().clamp_min(0.1)
            self.rolling_avg.add(f"loss/image_mixup", image_mixup_loss)
            loss += image_mixup_loss * self.mixup_weight

        sims = sims
        loss += self.contrast_loss(sims)
        self.rolling_avg.add(f"loss/total", loss)

        return loss

    def setup_hparams(self):
        recalls = ['A_r1', 'A_r5', 'A_r10', 'I_r1', 'I_r5', 'I_r10']

        if self.trainer.datamodule.use_extra_val_sets:
            datasets = ["Places", "AudioSet"]
        else:
            datasets = ["Val"]

        heads = ["total"]

        metric_names = [
            "hp/speech_basic_ap", "hp/speech_advanced_ap", "hp/sound_basic_ap",
            "hp/speech_basic_iou", "hp/speech_advanced_iou", "hp/sound_basic_iou",
        ]
        for dataset in datasets:
            for head in heads:
                for recall in recalls:
                    metric_names.append(f"hp/{dataset}/{head}/{recall}")

        if self.sim_agg_heads == 2:
            metric_names.extend(["hp/ap_dis", "hp/act_dis"])

        if hasattr(self.trainer, "datamodule"):
            all_hparams = {**self.hparams, **self.trainer.datamodule.hparams}
        else:
            all_hparams = self.hparams

        starting_values = {n: torch.nan for n in metric_names}
        self.logger.log_hyperparams(all_hparams, starting_values)

    def on_train_start(self):
        self.setup_hparams()
        self.hparams_logged = True

    def on_train_batch_start(self, batch, batch_idx):
        remake_optimizers = False

        if isinstance(self.image_aligner, ProgressiveGrowing):
            should_remake = self.image_aligner.maybe_change_phase(self.global_step)
            remake_optimizers = remake_optimizers or should_remake
        if isinstance(self.audio_aligner, ProgressiveGrowing):
            should_remake = self.audio_aligner.maybe_change_phase(self.global_step)
            remake_optimizers = remake_optimizers or should_remake

        if remake_optimizers:
            raise NotImplementedError()

    def _combine_preds(self, all_preds):
        temp = {}
        new_preds = {}

        # Collect tensors for each key into lists
        for d in all_preds:
            for key, value in d.items():
                if isinstance(value, torch.Tensor):
                    if key not in temp:
                        temp[key] = []
                    temp[key].append(value)

        # Concatenate all tensors for each key using a single call to torch.cat
        for key, tensor_list in temp.items():
            new_preds[key] = torch.cat(tensor_list)
        return new_preds

    def training_step(self, batch, batch_idx):
        assert batch[IMAGE_INPUT].shape[1] == 1

        preds = self.forward(batch)
        if DATA_SOURCE in batch:
            preds[DATA_SOURCE] = batch[DATA_SOURCE]

        if self.trainer.world_size > 1 and self.gather_tensors:
            for k, v in preds.items():
                new_v = v.contiguous()
                preds[k] = torch.cat(GatherLayer.apply(new_v), dim=0)

        if self.memory_buffer_size > 0:
            new_preds = self._combine_preds(list(self.memory_buffer) + [preds])
        else:
            new_preds = preds

        loss = self.loss(new_preds)

        if self.memory_buffer_size > 0:
            self.memory_buffer.append(self._recursive_detach(preds, gather=False))

        if self.trainer.is_global_zero and self.global_step % 50 == 1:
            writer = self.logger.experiment
            self.rolling_avg.logall(lambda k, v: writer.add_scalar(k, v, global_step=self.global_step))

        if self.trainer.scaler is not None:
            self.log("loss_scale", self.trainer.scaler.get_scale())

        if self.global_step % 10000 == 0 and self.global_step > 0:
            print("RESETTING TFEVENT FILE")
            self.logger.experiment.close()
            self.logger.experiment._get_file_writer()

        return loss

    def on_validation_start(self) -> None:
        if not self.hparams_logged:
            self.setup_hparams()
            self.hparams_logged = True

    def _auto_gather(self, t):
        if t.dtype == torch.bool:
            t = t.to(torch.float)

        if self.trainer.num_devices == 1:
            return t.cpu()

        t = torch.clone(t).contiguous()
        if self.trainer.is_global_zero:
            gather_list = [torch.zeros_like(t) for _ in range(dist.get_world_size())]
            dist.gather(t, gather_list)
            return torch.cat(gather_list, dim=0).cpu()
        else:
            dist.gather(t)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):

        with torch.no_grad():
            preds = self.forward(batch)

            ret = {}
            for k in preds.keys():
                if k in preds:
                    ret[k] = self._auto_gather(preds[k])

            batch_keys = [IMAGE_INPUT, "spec", "semseg", "num_pixels_per_class", 'total_length']
            for k in batch_keys:
                if k in batch:
                    ret[k] = self._auto_gather(batch[k])

            if "metadata" in batch:
                if isinstance(batch["metadata"]["id"], torch.Tensor):
                    ret["id"] = self._auto_gather(batch["metadata"]["id"])
                ret["index"] = self._auto_gather(batch["metadata"]["index"])

            return ret

    def _calc_recalls(self, sim):
        top_10_a = sim.topk(10, 0).indices == torch.arange(sim.shape[0]).unsqueeze(0)
        top_10_i = (sim.topk(10, 1).indices == torch.arange(sim.shape[0]).unsqueeze(1)).permute(1, 0)
        a_recall = lambda p: top_10_a[0:p].any(0).to(sim).mean()
        i_recall = lambda p: top_10_i[0:p].any(0).to(sim).mean()
        return {'A_r1': a_recall(1),
                'A_r5': a_recall(5),
                'A_r10': a_recall(10),
                'I_r1': i_recall(1),
                'I_r5': i_recall(5),
                'I_r10': i_recall(10)}

    def calc_recalls(self, preds, dataset):
        sim = self.sim_agg.forward_batched(
            preds=preds,
            agg_heads=False,
            batch_size=4,
        ).cpu()

        all_metrics = dict()
        for k, v in self._calc_recalls(sim.sum(-1)).items():
            all_metrics[f"hp/{dataset}/total/" + k] = v

        return all_metrics

    def retrieval_validation(self, outputs, dataset_name):
        if len(outputs) == 0:
            return

        if self.trainer.is_global_zero:
            results = flatten_preds(outputs)
            if not self.trainer.sanity_checking:
                print(results[IMAGE_FEATS].shape[0])
                # assert (results[IMAGE_FEATS].shape[0] == 1000)
            results[IMAGE_FEATS] = results[IMAGE_FEATS].cpu()
            results[AUDIO_FEATS] = results[AUDIO_FEATS].cuda()
            if self.sim_use_cls:
                results[AUDIO_CLS] = results[AUDIO_CLS].cuda()
                results[AUDIO_CLS] = results[AUDIO_CLS].cuda()

            results[AUDIO_MASK] = results[AUDIO_MASK].cuda()

            recalls = self.calc_recalls(results, dataset_name)

            results[IMAGE_FEATS] = results[IMAGE_FEATS].cuda()

            writer = self.logger.experiment
            print("here")
            for name, v in recalls.items():
                writer.add_scalar(f"{name}", v, self.global_step + 1)

    def semseg_validation(self, speech_preds, sound_preds):

        if self.trainer.is_global_zero:
            from eval_utils import get_paired_heatmaps
            def prep_preds(preds, loader):
                results = flatten_preds(preds)
                metadata = loader.dataset.metadata
                ordered_metadata = metadata.iloc[results["index"].numpy(), :].copy()
                ordered_metadata["order"] = range(len(ordered_metadata))
                return results, ordered_metadata

            [_, _, speech_loader, sound_loader] = self.trainer.val_dataloaders
            speech_results, speech_metadata = prep_preds(speech_preds, speech_loader)
            sound_results, sound_metadata = prep_preds(sound_preds, sound_loader)

            self.sound_metrics, unique_sound_indices = get_paired_heatmaps(
                self, sound_results, sound_metadata["ade_class_id"], None)

            self.speech_metrics, unique_word_indices = get_paired_heatmaps(
                self, speech_results, speech_metadata["ade_class_id"], speech_metadata["timing"])

            writer = self.logger.experiment

            all_metrics = {
                **{"sound_" + k: v for k, v in self.sound_metrics.items()},
                **{"speech_" + k: v for k, v in self.speech_metrics.items()},
            }

            for k, v in all_metrics.items():
                writer.add_scalar(f"hp/{k}", torch.tensor(v).mean(), self.global_step + 1)

    def disentangle_validation(self, word_preds, sound_preds):

        if len(word_preds) == 0 or len(sound_preds) == 0:
            return

        if self.trainer.is_global_zero:
            word_preds = flatten_preds(word_preds)
            sound_preds = flatten_preds(sound_preds)

            word_scores = self.sim_agg.get_pairwise_sims(
                word_preds,
                raw=False,
                agg_sim=True,
                agg_heads=False,
            )

            sound_scores = self.sim_agg.get_pairwise_sims(
                sound_preds,
                raw=False,
                agg_sim=True,
                agg_heads=False,
            )

            all_scores = torch.cat([word_scores, sound_scores], dim=0)
            all_scores -= all_scores.min(dim=0, keepdim=True).values
            all_scores /= all_scores.max(dim=0, keepdim=True).values.clamp_min(.0001)

            is_words = torch.cat([
                torch.ones(word_scores.shape[0]),
                torch.zeros(sound_scores.shape[0])], dim=0).to(torch.bool)

            assert all_scores.shape[1] == 2
            ap_matrix = torch.zeros(2, 2)
            act_matrix = torch.zeros(2, 2)

            for head in range(2):
                # writer.add_histogram(f"h{head}_all_scores", all_scores[:, head])
                for dataset_num in range(2):
                    if dataset_num == 0:
                        labels = is_words
                    else:
                        labels = ~is_words

                    ap_matrix[head, dataset_num] = binary_average_precision(
                        all_scores[:, head].cpu(), labels.to(torch.int64).cpu())

                    act_matrix[head, dataset_num] = 1 - (all_scores[:, head][labels]).mean()

            ap_dis = max(.5 * (ap_matrix[0, 0] + ap_matrix[1, 1]),
                         .5 * (ap_matrix[0, 1] + ap_matrix[1, 0]))

            act_dis = max(.5 * (act_matrix[0, 0] + act_matrix[1, 1]),
                          .5 * (act_matrix[0, 1] + act_matrix[1, 0]))

            print("AP", ap_matrix)
            print("AP dis", ap_dis)
            print("Act", act_matrix)
            print("Act dis", act_dis)

            writer = self.logger.experiment
            writer.add_scalar("hp/ap_dis", ap_dis, self.global_step + 1)
            writer.add_scalar("hp/act_dis", act_dis, self.global_step + 1)

    def validation_epoch_end(self, outputs) -> None:
        print("Val end")
        with torch.no_grad():
            if self.trainer.datamodule.use_extra_val_sets:
                if self.sim_agg_heads == 2:
                    self.disentangle_validation(outputs[0], outputs[1])
                self.retrieval_validation(outputs[0], "Places")
                self.retrieval_validation(outputs[1], "AudioSet")
                self.semseg_validation(outputs[2], outputs[3])

            else:
                print("HERE!")
                self.retrieval_validation(outputs, "Val")

        writer = self.logger.experiment
        writer.flush()

    def _recursive_detach(self, obj, gather=True):
        if isinstance(obj, torch.Tensor):
            if gather:
                return self._auto_gather(obj)
            else:
                obj.detach()
        elif isinstance(obj, dict):
            return {k: self._recursive_detach(v, gather) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._recursive_detach(v, gather) for v in obj]
        else:
            return obj

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        with torch.no_grad():
            predictions = {}
            for k, v in batch.items():
                predictions[k] = self._recursive_detach(v)
            for k, v in self.forward(batch).items():
                predictions[k] = self._auto_gather(v)

            return predictions

    def _configure_optimizers(self, full_train, lr):
        params = [
            *self.audio_aligner.parameters(),
            *self.image_aligner.parameters(),
            *self.sim_cal.parameters(),
            *self.sim_agg.parameters()
        ]

        if (self.finetune_image_model or self.image_lora) and full_train:
            params.extend(self.image_model.parameters())

        if (self.finetune_audio_model or self.audio_lora) and full_train:
            params.extend(self.audio_model.parameters())

        if self.learn_audio_cls:
            params.append(self.audio_cls)

        last_epoch = self.global_step - 1
        if self.optimizer == "adam":
            opt = torch.optim.Adam(params, lr=lr, eps=1e-7)
        elif self.optimizer == "nadam":
            opt = torch.optim.NAdam(params, lr=lr, eps=1e-7)
        else:
            raise ValueError(f"Unknown optimizer {self.optimizer}")

        if self.lr_schedule == "sgdr":
            scheduler = CosineAnnealingWarmRestarts(
                opt, self.lr_cycle_length, 2, eta_min=lr * 2e-2, last_epoch=last_epoch)
        else:
            scheduler = LambdaLR(opt, lr_lambda=lambda step: 1.0, last_epoch=last_epoch)

        if self.lr_warmup > 0:
            warmup = LambdaLR(
                opt,
                lr_lambda=lambda step: min(max(float(step), 0.0) / self.lr_warmup, 1.0),
                last_epoch=last_epoch,
            )
            scheduler = SequentialLR(
                opt,
                schedulers=[warmup, scheduler],
                milestones=[self.lr_warmup],
                last_epoch=last_epoch)

        scheduler = {"scheduler": scheduler, "interval": "step"}

        return [opt], [scheduler]

    def configure_optimizers(self):
        if self.full_train:
            return self._configure_optimizers(self.full_train, self.lr)
        else:
            return self._configure_optimizers(self.full_train, self.pretrain_lr)


@hydra.main(config_path="configs", config_name="av_align.yaml", version_base=None)
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    seed_everything(cfg.seed, workers=True)

    exp_name = f"{cfg.resume_prefix}"

    if cfg.image_model_type == "dino8":
        patch_size = 8 * cfg.image_pool_width
    elif cfg.image_model_type == "cavmae":
        patch_size = 16 * cfg.image_pool_width
    elif cfg.image_model_type == "imagebind":
        patch_size = 16 * cfg.image_pool_width
    elif cfg.image_model_type == "clip":
        patch_size = 16 * cfg.image_pool_width
    elif cfg.image_model_type == "cavmae-mixed":
        patch_size = 16 * cfg.image_pool_width
    elif cfg.image_model_type == "dinov2":
        patch_size = 14 * cfg.image_pool_width
    else:
        raise ValueError(f"Unknown patch size for model {cfg.image_model_type}")

    datamodule = AVDataModule(
        dataset_name=cfg.dataset_name,
        load_size=cfg.load_size,
        image_aug=cfg.image_aug,
        audio_aug=cfg.audio_aug,
        extra_audio_masking=cfg.extra_audio_masking,
        audio_model_type=cfg.audio_model_type,
        pytorch_data_dir=cfg.pytorch_data_dir,
        use_cached_embs=cfg.use_cached_embs,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        audio_level=cfg.audio_level,
        neg_audio=cfg.neg_audio,
        use_original_val_set=not cfg.use_extra_val_sets,
        use_extra_val_sets=cfg.use_extra_val_sets,
        data_for_plotting=False,
        quad_mixup=cfg.quad_mixup,
        bg_mixup=cfg.bg_mixup,
        patch_mixup=cfg.patch_mixup,
        patch_size=patch_size
    )
    datamodule.maybe_unpack(remove_source=cfg.submitting_to_aml)

    aligner = create_model_from_cfg(LitAVAligner, cfg, {})

    if cfg.starting_weights is not None:
        loaded = torch.load(join(cfg.output_root, cfg.starting_weights), map_location='cpu')
        state = loaded["state_dict"]
        aligner.load_state_dict(state, strict=cfg.load_strict)
        del state
        del loaded

    if cfg.num_gpus > 1:
        # strategy = "ddp_sharded"  # _find_unused_parameters_true"
        strategy = "ddp"  # _find_unused_parameters_true"
    else:
        strategy = "auto"

    if cfg.dataset_name in {"places-audio", "mixed", "audio-set", "mixed-full"}:
        val_args = dict(check_val_every_n_epoch=2)
    elif cfg.dataset_name in {"dolphin"}:
        val_args = dict(check_val_every_n_epoch=5)
    else:
        val_args = dict(val_check_interval=10000)

    # val_args = dict(val_check_interval=1000)

    def maybe_get_ckpt(ckpt_dir):
        if cfg.auto_resume and os.path.exists(ckpt_dir):
            print(f"Attempting to resume from {ckpt_dir}")
            candidates = os.listdir(ckpt_dir)
            assert (len(candidates) == 1)
            return join(ckpt_dir, candidates[0])
        elif cfg.auto_resume:
            print(f"Could not find checkpoint at {ckpt_dir}")
            return None
        else:
            return None

    log_dir = join(cfg.output_root, "logs", cfg.grouping_name, exp_name)
    ckpt_dir = join(cfg.output_root, "checkpoints", cfg.grouping_name, exp_name)

    import gc
    torch.cuda.empty_cache()
    gc.collect()

    def run_exp(aligner, full_train):
        trainer_args = dict(
            accelerator='gpu',
            strategy=strategy,
            devices=cfg.num_gpus,
            num_sanity_val_steps=cfg.num_sanity_val_steps,
            log_every_n_steps=50,
            reload_dataloaders_every_n_epochs=10,
            precision="16",
            # profiler="simple",
            # precision="bf16",
            max_steps=cfg.max_steps,
            **val_args)

        aligner.set_full_train(full_train)
        if full_train:
            suffix = "train"
        else:
            suffix = "pretrain"
            trainer_args["max_steps"] = cfg.pretrain_steps

        print(f"Starting {suffix} phase")

        logger = TensorBoardLogger(join(log_dir, suffix), default_hp_metric=False)
        callbacks = [
            ModelCheckpoint(join(ckpt_dir, suffix), every_n_epochs=1),
            LearningRateMonitor(logging_interval='step'),
        ]
        Trainer(logger=logger,
                callbacks=callbacks,
                **trainer_args).fit(
            aligner,
            datamodule=datamodule,
            ckpt_path=maybe_get_ckpt(join(ckpt_dir, suffix)))

    train_chkpt = maybe_get_ckpt(join(ckpt_dir, "train"))

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if cfg.pretrain_steps > 0 and train_chkpt is None:
        run_exp(aligner, full_train=False)
    run_exp(aligner, full_train=True)


if __name__ == "__main__":
    my_app()
