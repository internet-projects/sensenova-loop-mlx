# Copyright © 2026 NetProjects (Paris Michalas). Apache-2.0.
# Qwen3.5 vision tower (as used by sensenova/SenseNova-Loop-9B) in MLX.
# Written from the PyTorch reference (transformers 5.18 models/qwen3_5/modeling_qwen3_5.py: Qwen3_5VisionModel and
# transformers/vision_utils.py helpers). Layout notes in vision-phase0/NOTES.md. Not affiliated with SenseTime / Qwen.
#
#   pixel_values [n_patches, 3*2*16*16] (processor order C, T, ph, pw; patches in 2x2 merge-block order)
#   -> patch embed (Conv3d with stride == kernel, done as one Linear)
#   -> + learned 48x48 position table, bilinear (align_corners=True) resampled to each image's patch grid
#   -> 27 x [LayerNorm, full attention within each frame + 2-D rope (h, w), LayerNorm, GELU-tanh MLP]
#   -> merger: LayerNorm per patch, 4 patches (2x2) concatenated, Linear, GELU (erf), Linear -> [n_tokens, 4096]
# No deepstack, no window attention.

import math
from typing import List, Sequence, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

VISION_PREFIX = "model.visual."


class VisionConfig:
    def __init__(self, d: dict):
        self.depth = int(d.get("depth", 27))
        self.hidden_size = int(d.get("hidden_size", 1152))
        self.intermediate_size = int(d.get("intermediate_size", 4304))
        self.num_heads = int(d.get("num_heads", 16))
        self.in_channels = int(d.get("in_channels", 3))
        self.patch_size = int(d.get("patch_size", 16))
        self.temporal_patch_size = int(d.get("temporal_patch_size", 2))
        self.spatial_merge_size = int(d.get("spatial_merge_size", 2))
        self.out_hidden_size = int(d.get("out_hidden_size", 4096))
        self.num_position_embeddings = int(d.get("num_position_embeddings", 2304))
        rp = d.get("rope_parameters") or {}
        self.rope_theta = float(rp.get("rope_theta", 10000.0))
        if d.get("hidden_act", "gelu_pytorch_tanh") != "gelu_pytorch_tanh":
            raise ValueError(f"unsupported vision hidden_act {d.get('hidden_act')}")
        if d.get("deepstack_visual_indexes"):
            raise ValueError("deepstack is not supported (SenseNova-Loop has none)")


def _gelu_tanh(x):
    return 0.5 * x * (1 + mx.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))


def _rotate_half(x):
    h = x.shape[-1] // 2
    return mx.concatenate([-x[..., h:], x[..., :h]], axis=-1)


class PatchEmbed(nn.Module):
    def __init__(self, c: VisionConfig):
        super().__init__()
        self.proj = nn.Linear(c.in_channels * c.temporal_patch_size * c.patch_size * c.patch_size, c.hidden_size)

    def __call__(self, x):
        return self.proj(x.astype(self.proj.weight.dtype))


class VisionAttention(nn.Module):
    def __init__(self, c: VisionConfig):
        super().__init__()
        self.num_heads = c.num_heads
        self.head_dim = c.hidden_size // c.num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(c.hidden_size, 3 * c.hidden_size)
        self.proj = nn.Linear(c.hidden_size, c.hidden_size)

    def __call__(self, x, cos, sin, seqlens: Sequence[int]):
        N = x.shape[0]
        qkv = self.qkv(x).reshape(N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        dt = q.dtype
        # rope in float32, then back to the activation dtype (as apply_rotary_pos_emb_vision)
        c_, s_ = cos[:, None, :], sin[:, None, :]
        qf, kf = q.astype(mx.float32), k.astype(mx.float32)
        q = (qf * c_ + _rotate_half(qf) * s_).astype(dt)
        k = (kf * c_ + _rotate_half(kf) * s_).astype(dt)
        q, k, v = (t.transpose(1, 0, 2)[None] for t in (q, k, v))  # [1, H, N, D]
        outs, start = [], 0
        for n in seqlens:  # full attention inside each frame, none across frames / images
            sl = slice(start, start + n)
            outs.append(mx.fast.scaled_dot_product_attention(q[:, :, sl], k[:, :, sl], v[:, :, sl], scale=self.scale))
            start += n
        o = outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=2)
        o = o[0].transpose(1, 0, 2).reshape(N, -1)
        return self.proj(o)


class VisionMLP(nn.Module):
    def __init__(self, c: VisionConfig):
        super().__init__()
        self.linear_fc1 = nn.Linear(c.hidden_size, c.intermediate_size)
        self.linear_fc2 = nn.Linear(c.intermediate_size, c.hidden_size)

    def __call__(self, x):
        return self.linear_fc2(_gelu_tanh(self.linear_fc1(x)))


class VisionBlock(nn.Module):
    def __init__(self, c: VisionConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(c.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(c.hidden_size, eps=1e-6)
        self.attn = VisionAttention(c)
        self.mlp = VisionMLP(c)

    def __call__(self, x, cos, sin, seqlens):
        x = x + self.attn(self.norm1(x), cos, sin, seqlens)
        return x + self.mlp(self.norm2(x))


class PatchMerger(nn.Module):
    def __init__(self, c: VisionConfig):
        super().__init__()
        self.hidden = c.hidden_size * c.spatial_merge_size**2
        self.norm = nn.LayerNorm(c.hidden_size, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.hidden, self.hidden)
        self.linear_fc2 = nn.Linear(self.hidden, c.out_hidden_size)

    def __call__(self, x):
        x = self.norm(x).reshape(-1, self.hidden)
        return self.linear_fc2(nn.gelu(self.linear_fc1(x)))


# ---- grid helpers (numpy; same math as transformers.vision_utils) ----

def _merge_order_rc(h: int, w: int, m: int) -> Tuple[np.ndarray, np.ndarray]:
    """(row, col) of every patch of an h x w patch grid, in 2x2 merge-block order (block row, block col, in row, in col)."""
    r, c = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    shape = (h // m, m, w // m, m)
    return r.reshape(shape).transpose(0, 2, 1, 3).reshape(-1), c.reshape(shape).transpose(0, 2, 1, 3).reshape(-1)


def _axis_taps(index: np.ndarray, size: int, side: int):
    """Bilinear, align_corners=True taps and weights (float32 math as the reference)."""
    idx = index.astype(np.float32)
    src = idx * np.float32(side - 1) / np.float32(max(size - 1, 1))
    fl = np.floor(src)
    off = np.arange(2, dtype=np.float32)
    taps = np.clip(fl.astype(np.int64)[:, None] + np.arange(2), 0, side - 1)
    dist = np.abs(src[:, None] - fl[:, None] - off)
    return taps, np.clip(1 - dist, 0, None).astype(np.float32)


def vision_grid_tables(grid_thw: Sequence[Sequence[int]], merge: int, side: int, head_dim: int, theta: float):
    """Per-patch position-table taps/weights, rope cos/sin and per-frame attention lengths for a list of (t, h, w)."""
    idx, wts, pos, seqlens = [], [], [], []
    for t, h, w in grid_thw:
        t, h, w = int(t), int(h), int(w)
        r, c = _merge_order_rc(h, w, merge)
        ht, hw_ = _axis_taps(r, h, side)
        wt, ww = _axis_taps(c, w, side)
        i4 = (ht[:, :, None] * side + wt[:, None, :]).reshape(-1, 4)
        w4 = (hw_[:, :, None] * ww[:, None, :]).reshape(-1, 4)
        idx.append(np.tile(i4, (t, 1)))
        wts.append(np.tile(w4, (t, 1)))
        pos.append(np.tile(np.stack([r, c], -1), (t, 1)))
        seqlens += [h * w] * t
    pos = np.concatenate(pos).astype(np.float32)
    spatial = head_dim // 2
    inv = (1.0 / (theta ** (np.arange(0, spatial, 2, dtype=np.float32) / spatial))).astype(np.float32)
    f = pos[:, :, None] * inv[None, None, :]  # [N, 2, spatial/2]
    f = np.concatenate([f[:, 0], f[:, 1]], -1)
    f = np.concatenate([f, f], -1)  # [N, head_dim]
    return (np.concatenate(idx), np.concatenate(wts), np.cos(f).astype(np.float32), np.sin(f).astype(np.float32),
            seqlens)


class VisionModel(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        c = self.config = VisionConfig(config)
        self.patch_embed = PatchEmbed(c)
        self.pos_embed = nn.Embedding(c.num_position_embeddings, c.hidden_size)
        self.side = int(c.num_position_embeddings**0.5)
        self.blocks = [VisionBlock(c) for _ in range(c.depth)]
        self.merger = PatchMerger(c)

    @property
    def spatial_merge_size(self):
        return self.config.spatial_merge_size

    def __call__(self, pixel_values: mx.array, grid_thw: Sequence[Sequence[int]]) -> mx.array:
        """pixel_values [n_patches, C*T*p*p] -> merged features [n_patches / 4, out_hidden_size]."""
        c = self.config
        idx, wts, cos, sin, seqlens = vision_grid_tables(grid_thw, c.spatial_merge_size, self.side,
                                                         c.hidden_size // c.num_heads, c.rope_theta)
        if sum(seqlens) != pixel_values.shape[0]:
            raise ValueError(f"grid_thw gives {sum(seqlens)} patches, pixel_values has {pixel_values.shape[0]}")
        h = self.patch_embed(pixel_values)
        pe = (self.pos_embed(mx.array(idx)) * mx.array(wts)[:, :, None]).sum(1)
        h = h + pe.astype(h.dtype)
        cos, sin = mx.array(cos), mx.array(sin)
        for blk in self.blocks:
            h = blk(h, cos, sin, seqlens)
        return self.merger(h)

    @staticmethod
    def sanitize(weights: dict) -> dict:
        """HF names (model.visual.* or already stripped) -> this module; Conv3d [O, C, T, p, p] -> Linear [O, C*T*p*p]."""
        out = {}
        for k, v in weights.items():
            if k.startswith(VISION_PREFIX):
                k = k[len(VISION_PREFIX):]
            elif k.startswith("vision_tower."):
                k = k[len("vision_tower."):]
            if k == "patch_embed.proj.weight" and v.ndim == 5:
                v = v.reshape(v.shape[0], -1)
            out[k] = v
        return out


def load_vision_weights(path_or_files, prefixes=(VISION_PREFIX, "vision_tower.")) -> dict:
    """Read only the vision tensors (keys starting with one of `prefixes`) from one or more safetensors files."""
    files = [path_or_files] if isinstance(path_or_files, (str, bytes)) or hasattr(path_or_files, "__fspath__") \
        else list(path_or_files)
    w = {}
    for f in files:
        for k, v in mx.load(str(f)).items():
            if k.startswith(tuple(prefixes)):
                w[k] = v
    return w


def load_vision_model(vision_config: dict, weights_file, dtype=None) -> "VisionModel":
    vm = VisionModel(vision_config)
    w = VisionModel.sanitize(load_vision_weights(weights_file))
    if dtype is not None:
        w = {k: v.astype(dtype) for k, v in w.items()}
    vm.load_weights(list(w.items()), strict=True)
    mx.eval(vm.parameters())
    return vm
