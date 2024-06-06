from functools import partial

import torch
import torch.nn.functional as F
from torch.nn import ModuleList

from denseav.featurizers.DINO import Block


class ChannelNorm(torch.nn.Module):

    def __init__(self, dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.norm = torch.nn.LayerNorm(dim, eps=1e-4)

    def forward_spatial(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

    def forward(self, x, cls):
        return self.forward_spatial(x), self.forward_cls(cls)

    def forward_cls(self, cls):
        if cls is not None:
            return self.norm(cls)
        else:
            return None


def id_conv(dim, strength=.9):
    conv = torch.nn.Conv2d(dim, dim, 1, padding="same")
    start_w = conv.weight.data
    conv.weight.data = torch.nn.Parameter(
        torch.eye(dim, device=start_w.device).unsqueeze(-1).unsqueeze(-1) * strength + start_w * (1 - strength))
    conv.bias.data = torch.nn.Parameter(conv.bias.data * (1 - strength))
    return conv


class LinearAligner(torch.nn.Module):
    def __init__(self, in_dim, out_dim, use_norm=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        if use_norm:
            self.norm = ChannelNorm(in_dim)
        else:
            self.norm = Identity2()

        if in_dim == out_dim:
            self.layer = id_conv(in_dim, 0)
        else:
            self.layer = torch.nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1)

        self.cls_layer = torch.nn.Linear(in_dim, out_dim)

    def forward(self, spatial, cls):
        norm_spatial, norm_cls = self.norm(spatial, cls)

        if cls is not None:
            aligned_cls = self.cls_layer(cls)
        else:
            aligned_cls = None

        return self.layer(norm_spatial), aligned_cls

class IdLinearAligner(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        assert self.out_dim == self.in_dim
        self.layer = id_conv(in_dim, 1.0)
    def forward(self, spatial, cls):
        return self.layer(spatial), cls


class FrequencyAvg(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, spatial, cls):
        return spatial.mean(2, keepdim=True), cls


class LearnedTimePool(torch.nn.Module):
    def __init__(self, dim, width, maxpool):
        super().__init__()
        self.dim = dim
        self.width = width
        self.norm = ChannelNorm(dim)
        if maxpool:
            self.layer = torch.nn.Sequential(
                torch.nn.Conv2d(dim, dim, kernel_size=width, stride=1, padding="same"),
                torch.nn.MaxPool2d(kernel_size=(1, width), stride=(1, width))
            )
        else:
            self.layer = torch.nn.Conv2d(dim, dim, kernel_size=(1, width), stride=(1, width))

    def forward(self, spatial, cls):
        norm_spatial, norm_cls = self.norm(spatial, cls)
        return self.layer(norm_spatial), norm_cls


class LearnedTimePool2(torch.nn.Module):
    def __init__(self, in_dim, out_dim, width, maxpool, use_cls_layer):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.width = width

        if maxpool:
            self.layer = torch.nn.Sequential(
                torch.nn.Conv2d(in_dim, out_dim, kernel_size=width, stride=1, padding="same"),
                torch.nn.MaxPool2d(kernel_size=(1, width), stride=(1, width))
            )
        else:
            self.layer = torch.nn.Conv2d(in_dim, out_dim, kernel_size=(1, width), stride=(1, width))

        self.use_cls_layer = use_cls_layer
        if use_cls_layer:
            self.cls_layer = torch.nn.Linear(in_dim, out_dim)

    def forward(self, spatial, cls):

        if cls is not None:
            if self.use_cls_layer:
                aligned_cls = self.cls_layer(cls)
            else:
                aligned_cls = cls
        else:
            aligned_cls = None

        return self.layer(spatial), aligned_cls


class Sequential2(torch.nn.Module):

    def __init__(self, *modules):
        super().__init__()
        self.mod_list = ModuleList(modules)

    def forward(self, x, y):
        results = (x, y)
        for m in self.mod_list:
            results = m(*results)
        return results


class ProgressiveGrowing(torch.nn.Module):

    def __init__(self, stages, phase_lengths):
        super().__init__()
        self.stages = torch.nn.ModuleList(stages)
        self.phase_lengths = torch.tensor(phase_lengths)
        assert len(self.phase_lengths) + 1 == len(self.stages)
        self.phase_boundaries = self.phase_lengths.cumsum(0)
        self.register_buffer('phase', torch.tensor([1]))

    def maybe_change_phase(self, global_step):
        needed_phase = (global_step >= self.phase_boundaries).to(torch.int64).sum().item() + 1
        if needed_phase != self.phase.item():
            print(f"Changing aligner phase to {needed_phase}")
            self.phase.copy_(torch.tensor([needed_phase]).to(self.phase.device))
            return True
        else:
            return False

    def parameters(self, recurse: bool = True):
        phase = self.phase.item()
        used_stages = self.stages[:phase]
        print(f"Progressive Growing at stage {phase}")
        all_params = []
        for stage in used_stages:
            all_params.extend(stage.parameters(recurse))
        return iter(all_params)

    def forward(self, spatial, cls):
        pipeline = Sequential2(*self.stages[:self.phase.item()])
        return pipeline(spatial, cls)


class Identity2(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return x, y


class SelfAttentionAligner(torch.nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        self.num_heads = 6
        if dim % self.num_heads != 0:
            self.padding = self.num_heads - (dim % self.num_heads)
        else:
            self.padding = 0

        self.block = Block(
            dim + self.padding,
            num_heads=self.num_heads,
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            drop=0.0,
            attn_drop=0.0,
            drop_path=0.0,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-4))

    def forward(self, spatial, cls):
        padded_feats = F.pad(spatial, [0, 0, 0, 0, self.padding, 0])

        B, C, H, W = padded_feats.shape
        proj_feats = padded_feats.reshape(B, C, H * W).permute(0, 2, 1)

        if cls is not None:
            assert len(cls.shape) == 2
            padded_cls = F.pad(cls, [self.padding, 0])
            proj_feats = torch.cat([padded_cls.unsqueeze(1), proj_feats], dim=1)

        aligned_feat, attn, qkv = self.block(proj_feats, return_qkv=True)

        if cls is not None:
            aligned_cls = aligned_feat[:, 0, :]
            aligned_spatial = aligned_feat[:, 1:, :]
        else:
            aligned_cls = None
            aligned_spatial = aligned_feat

        aligned_spatial = aligned_spatial.reshape(B, H, W, self.dim + self.padding).permute(0, 3, 1, 2)

        aligned_spatial = aligned_spatial[:, self.padding:, :, :]
        if aligned_cls is not None:
            aligned_cls = aligned_cls[:, self.padding:]

        return aligned_spatial, aligned_cls


def get_aligner(aligner_type, in_dim, out_dim, **kwargs):
    if aligner_type is None:
        return Identity2()

    if "prog" in aligner_type:
        phase_length = kwargs["phase_length"]

    if aligner_type == "image_linear":
        return LinearAligner(in_dim, out_dim)
    elif aligner_type == "image_idlinear":
        return IdLinearAligner(in_dim, out_dim)
    elif aligner_type == "image_linear_no_norm":
        return LinearAligner(in_dim, out_dim, use_norm=False)
    elif aligner_type == "image_id":
        return Identity2()
    elif aligner_type == "image_norm":
        return ChannelNorm(in_dim)
    elif aligner_type == "audio_linear":
        return Sequential2(
            LinearAligner(in_dim, out_dim),
            FrequencyAvg())
    elif aligner_type == "audio_sa":
        return Sequential2(
            LinearAligner(in_dim, out_dim),
            FrequencyAvg(),
            SelfAttentionAligner(out_dim)
        )
    elif aligner_type == "audio_sa_sa":
        return Sequential2(
            FrequencyAvg(),
            LinearAligner(in_dim, out_dim),
            SelfAttentionAligner(out_dim),
            SelfAttentionAligner(out_dim)
        )
    elif aligner_type == "audio_3_3_pool":
        return Sequential2(
            LinearAligner(in_dim, out_dim),
            FrequencyAvg(),
            LearnedTimePool(out_dim, 3, False),
            LearnedTimePool(out_dim, 3, False),
        )
    elif aligner_type == "audio_sa_3_3_pool":
        return Sequential2(
            LinearAligner(in_dim, out_dim),
            FrequencyAvg(),
            LearnedTimePool(out_dim, 3, False),
            LearnedTimePool(out_dim, 3, False),
            SelfAttentionAligner(out_dim)
        )
    elif aligner_type == "audio_sa_3_3_pool_2":
        return Sequential2(
            FrequencyAvg(),
            ChannelNorm(in_dim),
            LearnedTimePool2(in_dim, out_dim, 3, False, True),
            LearnedTimePool2(out_dim, out_dim, 3, False, False),
            SelfAttentionAligner(out_dim)
        )
    else:
        raise ValueError(f"Unknown aligner type {aligner_type}")
