from abc import abstractmethod

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from constants import *


@torch.jit.script
def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int):
    mask = mask.to(x)
    return (x * mask).sum(dim, keepdim=True) / mask.sum(dim, keepdim=True).clamp_min(.001)


@torch.jit.script
def masked_max(x: torch.Tensor, mask: torch.Tensor, dim: int):
    mask = mask.to(torch.bool)
    eps = 1e7
    # eps = torch.finfo(x.dtype).max
    return (x - (~mask) * eps).max(dim, keepdim=True).values


def masked_lse(x: torch.Tensor, mask: torch.Tensor, dim: int, temp):
    x = x.to(torch.float32)
    mask = mask.to(torch.float32)
    x_masked = (x - (1 - mask) * torch.finfo(x.dtype).max)
    return (torch.logsumexp(x_masked * temp, dim, keepdim=True) - torch.log(mask.sum(dim, keepdim=True))) / temp


class BaseAggregator(torch.nn.Module):

    def __init__(self, nonneg_sim, mask_silence, num_heads, head_agg, use_cls):
        super().__init__()

        self.nonneg_sim = nonneg_sim
        self.mask_silence = mask_silence
        self.num_heads = num_heads
        self.head_agg = head_agg
        self.use_cls = use_cls

    @abstractmethod
    def _agg_sim(self, sim, mask):
        pass

    def prepare_sims(self, sim, mask, agg_sim, agg_heads):
        sim_size = sim.shape
        assert len(mask.shape) == 2
        assert len(sim_size) in {6, 7}, f"sim has wrong number of dimensions: {sim.shape}"
        pairwise = len(sim_size) == 6

        if self.mask_silence:
            mask = mask
        else:
            mask = torch.ones_like(mask)

        if self.nonneg_sim:
            sim = sim.clamp_min(0)

        if pairwise:
            head_dim = 1
        else:
            head_dim = 2

        if self.head_agg == "max_elementwise" and agg_heads:
            sim = sim.max(head_dim, keepdim=True).values

        if agg_sim:
            sim = self._agg_sim(sim, mask)

        if agg_heads:
            if self.head_agg == "sum" or self.head_agg == "max_elementwise":
                sim = sim.sum(head_dim)
            elif self.head_agg == "max":
                sim = sim.max(head_dim).values
            else:
                raise ValueError(f"Unknown head_agg: {self.head_agg}")

        return sim

    def _get_full_sims(self, preds, raw, agg_sim, agg_heads):
        if agg_sim or agg_heads or raw:
            assert (agg_sim or agg_heads) != raw, "Cannot have raw on at the same time as agg_sim or agg_heads"

        audio_feats = preds[AUDIO_FEATS]
        audio_mask = preds[AUDIO_MASK]
        image_feats = preds[IMAGE_FEATS]

        b1, c2, f, t1 = audio_feats.shape
        b2, t2 = audio_mask.shape
        d, c1, h, w = image_feats.shape
        assert b1 == b2 and c1 == c2 and t1 == t2
        assert c1 % self.num_heads == 0
        new_c = c1 // self.num_heads
        audio_feats = audio_feats.reshape(b1, self.num_heads, new_c, f, t1)
        image_feats = image_feats.reshape(d, self.num_heads, new_c, h, w)
        raw_sims = torch.einsum(
            "akcft,vkchw->avkhwft",
            audio_feats.to(torch.float32),
            image_feats.to(torch.float32))

        if self.use_cls:
            audio_cls = preds[AUDIO_CLS].reshape(b1, self.num_heads, new_c)
            image_cls = preds[IMAGE_CLS].reshape(d, self.num_heads, new_c)
            cls_sims = torch.einsum(
                "akc,vkc->avk",
                audio_cls.to(torch.float32),
                image_cls.to(torch.float32))
            raw_sims += cls_sims.reshape(b1, d, self.num_heads, 1, 1, 1, 1)

        if raw:
            return raw_sims
        else:
            return self.prepare_sims(raw_sims, audio_mask, agg_sim, agg_heads)

    def get_pairwise_sims(self, preds, raw, agg_sim, agg_heads):
        if agg_sim or agg_heads or raw:
            assert (agg_sim or agg_heads) != raw, "Cannot have raw on at the same time as agg_sim or agg_heads"

        audio_feats = preds[AUDIO_FEATS]
        audio_mask = preds[AUDIO_MASK]
        image_feats = preds[IMAGE_FEATS]

        a1, c1, f, t1 = audio_feats.shape
        a2, t2 = audio_mask.shape

        assert c1 % self.num_heads == 0
        new_c = c1 // self.num_heads
        audio_feats = audio_feats.reshape(a1, self.num_heads, new_c, f, t1)

        if len(image_feats.shape) == 5:
            print("Using similarity for video, should only be called during plotting")
            v, vt, c2, h, w = image_feats.shape
            image_feats = image_feats.reshape(v, vt, self.num_heads, new_c, h, w)
            raw_sims = torch.einsum(
                "bkcft,bskchw,bt->bskhwft",
                audio_feats.to(torch.float32),
                image_feats.to(torch.float32),
                audio_mask.to(torch.float32))

            if self.use_cls:
                audio_cls = preds[AUDIO_CLS].reshape(v, self.num_heads, new_c)
                image_cls = preds[IMAGE_CLS].reshape(v, vt, self.num_heads, new_c)
                cls_sims = torch.einsum(
                    "bkc,bskc->bsk",
                    audio_cls.to(torch.float32),
                    image_cls.to(torch.float32))
                raw_sims += cls_sims.reshape(v, vt, self.num_heads, 1, 1, 1, 1)


        elif len(image_feats.shape) == 4:
            v, c2, h, w = image_feats.shape
            image_feats = image_feats.reshape(v, self.num_heads, new_c, h, w)
            raw_sims = torch.einsum(
                "bkcft,bkchw,bt->bkhwft",
                audio_feats.to(torch.float32),
                image_feats.to(torch.float32),
                audio_mask.to(torch.float32))

            if self.use_cls:
                audio_cls = preds[AUDIO_CLS].reshape(v, self.num_heads, new_c)
                image_cls = preds[IMAGE_CLS].reshape(v, self.num_heads, new_c)
                cls_sims = torch.einsum(
                    "bkc,bkc->bk",
                    audio_cls.to(torch.float32),
                    image_cls.to(torch.float32))
                raw_sims += cls_sims.reshape(v, self.num_heads, 1, 1, 1, 1)
        else:
            raise ValueError(f"Improper image shape: {image_feats.shape}")

        assert a1 == a2 and c2 == c2 and t1 == t2

        if raw:
            return raw_sims
        else:
            return self.prepare_sims(raw_sims, audio_mask, agg_sim, agg_heads)

    def forward(self, preds, agg_heads):
        return self._get_full_sims(
            preds, raw=False, agg_sim=True, agg_heads=agg_heads)

    def forward_batched(self, preds, agg_heads, batch_size):
        new_preds = {k: v for k, v in preds.items()}
        big_image_feats = new_preds.pop(IMAGE_FEATS)
        if self.use_cls:
            big_image_cls = new_preds.pop(IMAGE_CLS)

        n = big_image_feats.shape[0]
        n_steps = math.ceil(n / batch_size)
        outputs = []
        for step in tqdm(range(n_steps), "Calculating Sim", leave=False):
            new_preds[IMAGE_FEATS] = big_image_feats[step * batch_size:(step + 1) * batch_size].cuda()
            if self.use_cls:
                new_preds[IMAGE_CLS] = big_image_cls[step * batch_size:(step + 1) * batch_size].cuda()

            sim = self.forward(new_preds, agg_heads=agg_heads)
            outputs.append(sim.cpu())
        return torch.cat(outputs, dim=1)


class ImageThenAudioAggregator(BaseAggregator):

    def __init__(self, image_agg_type, audio_agg_type, nonneg_sim, mask_silence, num_heads, head_agg, use_cls):
        super().__init__(nonneg_sim, mask_silence, num_heads, head_agg, use_cls)
        if image_agg_type == "max":
            self.image_agg = lambda x, dim: x.max(dim=dim, keepdim=True).values
        elif image_agg_type == "avg":
            self.image_agg = lambda x, dim: x.mean(dim=dim, keepdim=True)
        else:
            raise ValueError(f"Unknown image_agg_type {image_agg_type}")

        if audio_agg_type == "max":
            self.time_agg = masked_max
        elif audio_agg_type == "avg":
            self.time_agg = masked_mean
        else:
            raise ValueError(f"Unknown audio_agg_type {audio_agg_type}")

        self.freq_agg = lambda x, dim: x.mean(dim=dim, keepdim=True)

    def _agg_sim(self, sim, mask):
        sim_shape = sim.shape
        new_mask_shape = [1] * len(sim_shape)
        new_mask_shape[0] = sim_shape[0]
        new_mask_shape[-1] = sim_shape[-1]
        mask = mask.reshape(new_mask_shape)
        sim = self.image_agg(sim, -3)
        sim = self.image_agg(sim, -4)
        sim = self.freq_agg(sim, -2)
        sim = self.time_agg(sim, mask, -1)
        return sim.squeeze(-1).squeeze(-1).squeeze(-1).squeeze(-1)


class PairedAggregator(BaseAggregator):

    def __init__(self, nonneg_sim, mask_silence, num_heads, head_agg, use_cls):
        super().__init__(nonneg_sim, mask_silence, num_heads, head_agg, use_cls)
        self.image_agg_max = lambda x, dim: x.max(dim=dim, keepdim=True).values
        self.image_agg_mean = lambda x, dim: x.mean(dim=dim, keepdim=True)

        self.time_agg_max = masked_max
        self.time_agg_mean = masked_mean

        self.freq_agg = lambda x, dim: x.mean(dim=dim, keepdim=True)

    def _agg_sim(self, sim, mask):
        sim_shape = sim.shape
        new_mask_shape = [1] * len(sim_shape)
        new_mask_shape[0] = sim_shape[0]
        new_mask_shape[-1] = sim_shape[-1]
        mask = mask.reshape(new_mask_shape)

        sim_1 = self.image_agg_max(sim, -3)
        sim_1 = self.image_agg_max(sim_1, -4)
        sim_1 = self.freq_agg(sim_1, -2)
        sim_1 = self.time_agg_mean(sim_1, mask, -1)

        sim_2 = self.freq_agg(sim, -2)
        sim_2 = self.time_agg_max(sim_2, mask, -1)
        sim_2 = self.image_agg_mean(sim_2, -3)
        sim_2 = self.image_agg_mean(sim_2, -4)

        sim = 1 / 2 * (sim_1 + sim_2)

        return sim.squeeze(-1).squeeze(-1).squeeze(-1).squeeze(-1)



class CAVMAEAggregator(BaseAggregator):

    def __init__(self, *args, **kwargs):
        super().__init__(False, False, 1, "sum", False)

    def _get_full_sims(self, preds, raw, agg_sim, agg_heads):
        if agg_sim:
            audio_feats = preds[AUDIO_FEATS]
            image_feats = preds[IMAGE_FEATS]
            pool_audio_feats = F.normalize(audio_feats.mean(dim=[-1, -2]), dim=1)
            pool_image_feats = F.normalize(image_feats.mean(dim=[-1, -2]), dim=1)
            sims = torch.einsum(
                "bc,dc->bd",
                pool_audio_feats.to(torch.float32),
                pool_image_feats.to(torch.float32))
            if agg_heads:
                return sims
            else:
                return sims.unsqueeze(-1)

        else:
            return BaseAggregator._get_full_sims(self, preds, raw, agg_sim, agg_heads)

    def get_pairwise_sims(self, preds, raw, agg_sim, agg_heads):
        if agg_sim:
            audio_feats = preds[AUDIO_FEATS]
            image_feats = preds[IMAGE_FEATS]
            pool_audio_feats = F.normalize(audio_feats.mean(dim=[-1, -2]), dim=1)
            pool_image_feats = F.normalize(image_feats.mean(dim=[-1, -2]), dim=1)
            sims = torch.einsum(
                "bc,bc->b",
                pool_audio_feats.to(torch.float32),
                pool_image_feats.to(torch.float32))
            if agg_heads:
                return sims
            else:
                return sims.unsqueeze(-1)

        else:
            return BaseAggregator.get_pairwise_sims(self, preds, raw, agg_sim, agg_heads)


class ImageBindAggregator(BaseAggregator):

    def __init__(self, num_heads, *args, **kwargs):
        super().__init__(False, False, num_heads, "sum", False)

    def _get_full_sims(self, preds, raw, agg_sim, agg_heads):
        if agg_sim:
            sims = torch.einsum(
                "bc,dc->bd",
                preds[AUDIO_CLS].to(torch.float32),
                preds[IMAGE_CLS].to(torch.float32))
            if agg_heads:
                return sims
            else:
                sims = sims.unsqueeze(-1)
                return sims.repeat(*([1] * (sims.dim() - 1)), self.num_heads)


        else:
            return BaseAggregator._get_full_sims(self, preds, raw, agg_sim, agg_heads)

    def get_pairwise_sims(self, preds, raw, agg_sim, agg_heads):
        if agg_sim:
            sims = torch.einsum(
                "bc,dc->b",
                preds[AUDIO_CLS].to(torch.float32),
                preds[IMAGE_CLS].to(torch.float32))
            if agg_heads:
                return sims
            else:
                sims = sims.unsqueeze(-1)
                return sims.repeat(*([1] * (sims.dim() - 1)), self.num_heads)

        else:
            return BaseAggregator.get_pairwise_sims(self, preds, raw, agg_sim, agg_heads)

    def forward_batched(self, preds, agg_heads, batch_size):
        return self.forward(preds, agg_heads)


class SimPool(nn.Module):
    def __init__(self, dim, num_heads=1, qkv_bias=False, qk_scale=None, gamma=None, use_beta=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.norm_patches = nn.LayerNorm(dim, eps=1e-6)

        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)

        if gamma is not None:
            self.gamma = torch.tensor([gamma])
            if use_beta:
                self.beta = nn.Parameter(torch.tensor([0.0]))
        self.eps = torch.tensor([1e-6])

        self.gamma = gamma
        self.use_beta = use_beta

    def prepare_input(self, x):
        if len(x.shape) == 3:  # Transformer
            # Input tensor dimensions:
            # x: (B, N, d), where B is batch size, N are patch tokens, d is depth (channels)
            B, N, d = x.shape
            gap_cls = x.mean(-2)  # (B, N, d) -> (B, d)
            gap_cls = gap_cls.unsqueeze(1)  # (B, d) -> (B, 1, d)
            return gap_cls, x
        if len(x.shape) == 4:  # CNN
            # Input tensor dimensions:
            # x: (B, d, H, W), where B is batch size, d is depth (channels), H is height, and W is width
            B, d, H, W = x.shape
            gap_cls = x.mean([-2, -1])  # (B, d, H, W) -> (B, d)
            x = x.reshape(B, d, H * W).permute(0, 2, 1)  # (B, d, H, W) -> (B, d, H*W) -> (B, H*W, d)
            gap_cls = gap_cls.unsqueeze(1)  # (B, d) -> (B, 1, d)
            return gap_cls, x
        else:
            raise ValueError(f"Unsupported number of dimensions in input tensor: {len(x.shape)}")

    def forward(self, x):
        self.eps = self.eps.to(x.device)
        # Prepare input tensor and perform GAP as initialization
        gap_cls, x = self.prepare_input(x)

        # Prepare queries (q), keys (k), and values (v)
        q, k, v = gap_cls, self.norm_patches(x), self.norm_patches(x)

        # Extract dimensions after normalization
        Bq, Nq, dq = q.shape
        Bk, Nk, dk = k.shape
        Bv, Nv, dv = v.shape

        # Check dimension consistency across batches and channels
        assert Bq == Bk == Bv
        assert dq == dk == dv

        # Apply linear transformation for queries and keys then reshape
        qq = self.wq(q).reshape(Bq, Nq, self.num_heads, dq // self.num_heads).permute(0, 2, 1,
                                                                                      3)  # (Bq, Nq, dq) -> (B, num_heads, Nq, dq/num_heads)
        kk = self.wk(k).reshape(Bk, Nk, self.num_heads, dk // self.num_heads).permute(0, 2, 1,
                                                                                      3)  # (Bk, Nk, dk) -> (B, num_heads, Nk, dk/num_heads)

        vv = v.reshape(Bv, Nv, self.num_heads, dv // self.num_heads).permute(0, 2, 1,
                                                                             3)  # (Bv, Nv, dv) -> (B, num_heads, Nv, dv/num_heads)

        # Compute attention scores
        attn = (qq @ kk.transpose(-2, -1)) * self.scale
        # Apply softmax for normalization
        attn = attn.softmax(dim=-1)

        # If gamma scaling is used
        if self.gamma is not None:
            # Apply gamma scaling on values and compute the weighted sum using attention scores
            x = torch.pow(attn @ torch.pow((vv - vv.min() + self.eps), self.gamma),
                          1 / self.gamma)  # (B, num_heads, Nv, dv/num_heads) -> (B, 1, 1, d)
            # If use_beta, add a learnable translation
            if self.use_beta:
                x = x + self.beta
        else:
            # Compute the weighted sum using attention scores
            x = (attn @ vv).transpose(1, 2).reshape(Bq, Nq, dq)

        return x.squeeze()



class SimPoolAggregator(BaseAggregator):

    def __init__(self, num_heads, dim, *args, **kwargs):
        super().__init__(False, False, num_heads, "sum", False)
        self.pool = SimPool(dim, gamma=1.25)

    def _get_full_sims(self, preds, raw, agg_sim, agg_heads):
        if agg_sim:
            device = self.pool.wq.weight.data.device
            pooled_audio = self.pool(preds[AUDIO_FEATS].to(torch.float32).to(device))
            pooled_image = self.pool(preds[IMAGE_FEATS].to(torch.float32).to(device))

            sims = torch.einsum(
                "bc,dc->bd",
                pooled_audio,
                pooled_image)
            if agg_heads:
                return sims
            else:
                sims = sims.unsqueeze(-1)
                return sims.repeat(*([1] * (sims.dim() - 1)), self.num_heads)


        else:
            return BaseAggregator._get_full_sims(self, preds, raw, agg_sim, agg_heads)

    def get_pairwise_sims(self, preds, raw, agg_sim, agg_heads):
        if agg_sim:
            device = self.pool.wq.weight.data.device
            pooled_audio = self.pool(preds[AUDIO_FEATS].to(torch.float32).to(device))
            pooled_image = self.pool(preds[IMAGE_FEATS].to(torch.float32).to(device))

            sims = torch.einsum(
                "bc,dc->b",
                pooled_audio,
                pooled_image)
            if agg_heads:
                return sims
            else:
                sims = sims.unsqueeze(-1)
                return sims.repeat(*([1] * (sims.dim() - 1)), self.num_heads)

        else:
            return BaseAggregator.get_pairwise_sims(self, preds, raw, agg_sim, agg_heads)

    def forward_batched(self, preds, agg_heads, batch_size):
        return self.forward(preds, agg_heads)



def get_aggregator(sim_agg_type, nonneg_sim, mask_silence, num_heads, head_agg, use_cls, dim):
    shared_args = dict(
        nonneg_sim=nonneg_sim,
        mask_silence=mask_silence,
        num_heads=num_heads,
        head_agg=head_agg,
        use_cls=use_cls,
    )

    if sim_agg_type == "paired":
        agg1 = PairedAggregator(**shared_args)
    elif sim_agg_type == "misa":
        agg1 = ImageThenAudioAggregator("max", "avg", **shared_args)
    elif sim_agg_type == "mima":
        agg1 = ImageThenAudioAggregator("max", "max", **shared_args)
    elif sim_agg_type == "sisa":
        agg1 = ImageThenAudioAggregator("avg", "avg", **shared_args)
    elif sim_agg_type == "cavmae":
        agg1 = CAVMAEAggregator()
    elif sim_agg_type == "imagebind":
        agg1 = ImageBindAggregator(num_heads=shared_args["num_heads"])
    elif sim_agg_type == "simpool":
        agg1 = SimPoolAggregator(num_heads=shared_args["num_heads"], dim=dim)
    else:
        raise ValueError(f"Unknown loss_type {sim_agg_type}")

    return agg1

