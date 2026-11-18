# Copyright © 2026 NetProjects (Paris Michalas). Apache-2.0.
# MLX port of SenseNova-Loop (sensenova/SenseNova-Loop-9B @ 329a65f7 and SenseNova-Loop-9B-Preview @ 9937e91; both ship
# the same modeling_sensenova_loop.py). The release adds YaRN rope (512K), handled by mlx-lm's qwen3_5 attention.
# Not affiliated with SenseTime / SenseNova.
#
# SenseNova-Loop is Qwen3.5 (text) with two changes:
#   1. Looped execution: physical layers [start, end) run `eval_count` times with shared weights. Every execution
#      ("visit") has its own cache slot (KV for full attention, conv + recurrent state for Gated DeltaNet).
#   2. Two-stream mHC (manifold-constrained hyper-connections) on layers [mhc.start, mhc.end): the hidden state is
#      copied into `streams` residual streams at the first mHC layer and averaged back at the last one, on every pass.
#      Each mHC layer has two fp32 routers (mhc_attn, mhc_ffn) that give pre / post gates and a Sinkhorn mixing matrix.
#
# Works as mlx_lm/models/sensenova_loop.py (model_type "sensenova_loop") or as a custom model_file with
# trust_remote_code=True.

import os
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

try:
    from . import qwen3_5 as q35
    from .base import BaseModelArgs, create_attention_mask, create_ssm_mask, scaled_dot_product_attention
    from .cache import ArraysCache, KVCache
except ImportError:  # loaded as a model_file
    from mlx_lm.models import qwen3_5 as q35
    from mlx_lm.models.base import BaseModelArgs, create_attention_mask, create_ssm_mask, scaled_dot_product_attention
    from mlx_lm.models.cache import ArraysCache, KVCache

import numpy as np


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    text_config: dict
    loop_settings: dict = field(default_factory=dict)
    mhc: dict = field(default_factory=dict)
    # image input (sensenova_mlx/vision.py runs the vision tower; the text model only needs these ids)
    vision_config: Optional[dict] = None
    image_token_id: int = 248056
    video_token_id: int = 248057
    vision_start_token_id: int = 248053


def execution_plan(num_layers: int, loop: dict, mhc: dict) -> Tuple[int, ...]:
    if not loop:
        return tuple(range(num_layers))
    start, end = int(loop["start"]), int(loop["end"])
    count = int(loop.get("eval_count", (loop.get("counts") or [1])[0]))
    if "counts" in loop and list(loop["counts"]) != [count]:
        raise ValueError(f"loop_settings.counts {loop['counts']} must be [{count}]")
    if not (0 <= start < end <= num_layers) or count < 1:
        raise ValueError(f"bad loop_settings {loop}")
    if mhc:
        ms, me = int(mhc["start"]), int(mhc["end"])
        inside = (start <= ms < me <= end) or (me <= start) or (ms >= end)
        if not (0 <= ms < me <= num_layers) or not inside:
            raise ValueError(f"mhc range [{ms}, {me}) must sit inside one loop phase")
    return tuple(range(start)) + tuple(range(start, end)) * count + tuple(range(end, num_layers))


def sinkhorn(logits: mx.array, iterations: int) -> mx.array:
    """Log-space Sinkhorn: rows then columns, `iterations` times, then exp (fp32)."""
    logits = logits.astype(mx.float32)
    for _ in range(iterations):
        logits = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        logits = logits - mx.logsumexp(logits, axis=-2, keepdims=True)
    return mx.exp(logits)


class MHCRouter(nn.Module):
    """fp32 router. Kept as raw parameters (not nn.Linear) so convert never quantizes it."""

    def __init__(self, dim: int, streams: int, iterations: int, fixed_identity: bool, pre_scale: float):
        super().__init__()
        self.streams = streams
        self.iterations = iterations
        self.fixed_identity = fixed_identity
        self.pre_scale = pre_scale
        out = 2 * streams + (0 if fixed_identity else streams * streams)
        self.weight = mx.zeros((out, streams * dim), dtype=mx.float32)
        self.bias = mx.zeros((out,), dtype=mx.float32)

    def __call__(self, residual: mx.array):
        # residual: (..., streams, dim) in the model dtype
        fn = _route_fn(self.streams, self.iterations, self.fixed_identity, self.pre_scale)
        branch, post, mix = fn(residual, self.weight, self.bias)
        return branch, post, (None if self.fixed_identity else mix)

    @staticmethod
    def merge(residual: mx.array, branch: mx.array, post: mx.array, mix: Optional[mx.array]) -> mx.array:
        if mix is None:
            return residual + post.astype(branch.dtype)[..., None] * branch[..., None, :]
        return _merge_fn()(residual, branch, post, mix)


def _route(residual, weight, bias, s, iterations, fixed_identity, pre_scale):
    flat = residual.reshape(*residual.shape[:-2], -1).astype(mx.float32)
    normalized = flat * mx.rsqrt(mx.mean(mx.square(flat), axis=-1, keepdims=True) + 1e-6)
    logits = normalized @ weight.astype(mx.float32).T + bias.astype(mx.float32)
    pre = mx.sigmoid(logits[..., :s])
    post = 2 * mx.sigmoid(logits[..., s : 2 * s])
    if fixed_identity:
        mix = mx.zeros((1,), dtype=mx.float32)
    else:
        mix = sinkhorn(logits[..., 2 * s :].reshape(*logits.shape[:-1], s, s), iterations)
    branch = (pre.astype(residual.dtype)[..., None, :] @ residual).squeeze(-2)
    if pre_scale != 1.0:
        branch = branch * pre_scale
    return branch, post, mix


def _merge(residual, branch, post, mix):
    return mix.astype(residual.dtype) @ residual + post.astype(branch.dtype)[..., None] * branch[..., None, :]


# The router is ~45 tiny ops (20 Sinkhorn iterations) and runs 96 times per token, so it is compiled into one kernel
# graph; SN_NO_COMPILE=1 runs it eagerly (same math).
_COMPILE = os.environ.get("SN_NO_COMPILE", "0") != "1"
_ROUTE_CACHE: Dict[tuple, Any] = {}


def _route_fn(s, iterations, fixed_identity, pre_scale):
    key = (s, iterations, fixed_identity, pre_scale)
    if key not in _ROUTE_CACHE:
        f = partial(_route, s=s, iterations=iterations, fixed_identity=fixed_identity, pre_scale=pre_scale)
        _ROUTE_CACHE[key] = mx.compile(f) if _COMPILE else f
    return _ROUTE_CACHE[key]


_MERGE = []


def _merge_fn():
    if not _MERGE:
        _MERGE.append(mx.compile(_merge) if _COMPILE else _merge)
    return _MERGE[0]


class MHCDecoderLayer(q35.DecoderLayer):
    def __init__(self, args: q35.TextModelArgs, layer_idx: int, mhc: dict):
        super().__init__(args, layer_idx)
        self.mhc_start, self.mhc_end = int(mhc["start"]), int(mhc["end"])
        self.layer_idx = layer_idx
        self.streams = int(mhc["streams"])
        kw = dict(
            dim=args.hidden_size,
            streams=self.streams,
            iterations=int(mhc["iterations"]),
            fixed_identity=bool(mhc.get("fixed_identity", False)),
            pre_scale=float(mhc.get("pre_scale", 1.0)),
        )
        self.mhc_attn = MHCRouter(**kw)
        self.mhc_ffn = MHCRouter(**kw)

    def __call__(self, x: mx.array, mask=None, cache=None) -> mx.array:
        # enter: copy into streams at the first mHC layer, otherwise x already carries (B, L, streams, D)
        if self.layer_idx == self.mhc_start:
            residual = mx.broadcast_to(x[..., None, :], (*x.shape[:-1], self.streams, x.shape[-1]))
        else:
            residual = x
        branch, post, mix = self.mhc_attn(residual)
        branch = self.input_layernorm(branch)
        if self.is_linear:
            branch = self.linear_attn(branch, mask, cache)
        else:
            branch = self.self_attn(branch, mask, cache)
        residual = MHCRouter.merge(residual, branch, post, mix)
        branch, post, mix = self.mhc_ffn(residual)
        branch = self.mlp(self.post_attention_layernorm(branch))
        out = MHCRouter.merge(residual, branch, post, mix)
        # leave: average the streams after the last mHC layer
        return out.mean(axis=-2) if self.layer_idx == self.mhc_end - 1 else out


# ---- image input: 3-axis (t, h, w) multimodal rope ----
# Text-only prompts never activate any of this: the attention below falls straight through to mlx-lm's Qwen3.5
# attention (1-D rope at cache.offset), which equals interleaved mRoPE when all three position axes are equal.
# With an image, the prompt gets explicit [3, B, T] positions (get_rope_index) and every later token sits at
# cache offset + rope_delta. Positions -> cos/sin are computed once per forward and shared by all 56 visits, as in
# SenseNova's reference (SenseNovaLoopTextModel.forward computes position_embeddings once).


def get_rope_index(input_ids, image_grid_thw, image_token_id: int, spatial_merge_size: int = 2,
                   video_token_id: Optional[int] = None):
    """numpy port of transformers Qwen3_5Model.get_rope_index for images (no padding mask, no video).

    input_ids [B, T] -> (position_ids int64 [3, B, T], rope_deltas int64 [B]). Text runs advance all three axes by 1
    per token; an image run starting at position p gets t = p, h = p + row, w = p + col over the merged grid, and the
    next text resumes at p + max(gh, gw) / merge.
    """
    ids = np.asarray(input_ids)
    if ids.ndim == 1:
        ids = ids[None]
    if video_token_id is not None and (ids == video_token_id).any():
        raise NotImplementedError("video input is not supported")
    grids = iter([tuple(int(v) for v in g) for g in (image_grid_thw if image_grid_thw is not None else [])])
    B, T = ids.shape
    pos = np.zeros((3, B, T), dtype=np.int64)
    deltas = np.zeros((B,), dtype=np.int64)
    m = spatial_merge_size
    for b in range(B):
        is_img = ids[b] == image_token_id
        chunks, cur, i = [], 0, 0
        while i < T:
            j = i
            while j < T and is_img[j] == is_img[i]:
                j += 1
            if not is_img[i]:
                chunks.append(np.broadcast_to(np.arange(j - i) + cur, (3, j - i)))
                cur += j - i
            else:
                t, h, w = next(grids)
                gt, gh, gw = t, h // m, w // m
                if gt * gh * gw != j - i:
                    raise ValueError(f"image run of {j - i} tokens does not match grid {(t, h, w)}")
                tt, hh, ww = np.meshgrid(np.arange(gt), np.arange(gh), np.arange(gw), indexing="ij")
                chunks.append(np.stack([tt.reshape(-1) + cur, hh.reshape(-1) + cur, ww.reshape(-1) + cur]))
                cur += max(h, w) // m
            i = j
        p = np.concatenate(chunks, axis=1)
        pos[:, b] = p
        deltas[b] = p.max() + 1 - T
    return pos, deltas


class MRoPEContext:
    """Shared, per-model state for an image prompt: positions of the prompt and the decode delta (plain object, so
    it is never part of the module's parameters)."""

    def __init__(self, section=(11, 11, 10), interleaved=True):
        self.section, self.interleaved = list(section), bool(interleaved)
        self.position_ids = None  # mx int32 [3, B, T] for the prompt
        self.delta = None  # mx int32 [B]
        self.cos = self.sin = None  # per-forward cos/sin [B, 1, L, rot_dims] (float32)

    @property
    def active(self):
        return self.position_ids is not None


def _mrope_axis_map(half: int, section, interleaved: bool) -> np.ndarray:
    axis = np.zeros((half,), dtype=np.int64)
    if interleaved:  # Qwen3_5TextRotaryEmbedding.recomposition_frequencies
        for d, off in ((1, 1), (2, 2)):
            axis[off : section[d] * 3 : 3] = d
    else:
        bounds = np.cumsum([0] + list(section))
        for d in range(3):
            axis[bounds[d] : bounds[d + 1]] = d
    return axis


class MRoPEAttention(q35.Attention):
    """mlx-lm Qwen3NextAttention with an explicit-position rope when an image prompt is active (body adapted from
    mlx-lm, MIT). Inactive -> the parent __call__ unchanged."""

    _mrope_ctx: Optional[MRoPEContext] = None

    def __call__(self, x: mx.array, mask=None, cache=None) -> mx.array:
        ctx = self._mrope_ctx
        if ctx is None or ctx.cos is None:
            return super().__call__(x, mask, cache)
        B, L, D = x.shape
        queries, gate = mx.split(self.q_proj(x).reshape(B, L, self.num_attention_heads, -1), 2, axis=-1)
        gate = gate.reshape(B, L, -1)
        keys, values = self.k_proj(x), self.v_proj(x)
        queries = self.q_norm(queries).transpose(0, 2, 1, 3)
        keys = self.k_norm(keys.reshape(B, L, self.num_key_value_heads, -1)).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)
        queries = _apply_mrope(queries, ctx.cos, ctx.sin)
        keys = _apply_mrope(keys, ctx.cos, ctx.sin)
        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)
        output = scaled_dot_product_attention(queries, keys, values, cache=cache, scale=self.scale, mask=mask)
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output * mx.sigmoid(gate))


def _apply_mrope(x, cos, sin):
    """Non-traditional (rotate-half) rope on the first cos.shape[-1] dims; yarn mscale is already folded into cos/sin."""
    rd = cos.shape[-1]
    xr = x[..., :rd].astype(mx.float32)
    h = rd // 2
    rot = mx.concatenate([-xr[..., h:], xr[..., :h]], axis=-1)
    out = (xr * cos + rot * sin).astype(x.dtype)
    return mx.concatenate([out, x[..., rd:]], axis=-1) if rd < x.shape[-1] else out


def _rope_inv_freq_and_scale(rope) -> Tuple[mx.array, float]:
    """Inverse frequencies and output scale of an mlx-lm rope module (YarnRoPE / nn.RoPE)."""
    if hasattr(rope, "_freqs"):
        inv = 1.0 / rope._freqs.astype(mx.float32)
    else:  # nn.RoPE: theta_i = scale * pos * base^(-2i/dims)
        dims, base, scale = rope.dims, rope.base, getattr(rope, "scale", 1.0)
        inv = scale / (base ** (mx.arange(0, dims, 2, dtype=mx.float32) / dims))
    return inv, float(getattr(rope, "mscale", 1.0))


class _CastRMSNorm(nn.RMSNorm):
    """RMSNorm whose output keeps the input dtype. With bf16 weights it is identical to nn.RMSNorm; with fp32 weights
    (the checkpoint's 1 + w kept in float32, as Transformers computes it) the product is formed in fp32 and cast back."""

    def __call__(self, x):
        return mx.fast.rms_norm(x, self.weight, self.eps).astype(x.dtype)


def _use_cast_norms(module) -> None:
    for _, m in module.named_modules():
        if type(m) is nn.RMSNorm:
            m.__class__ = _CastRMSNorm


class SenseNovaLoopTextModel(nn.Module):
    def __init__(self, args: q35.TextModelArgs, loop: dict, mhc: dict):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        ms, me = (int(mhc["start"]), int(mhc["end"])) if mhc else (0, 0)
        self.layers = [
            MHCDecoderLayer(args, i, mhc) if ms <= i < me else q35.DecoderLayer(args, i)
            for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.plan = execution_plan(args.num_hidden_layers, loop, mhc)
        self.fa_visit = next((e for e, p in enumerate(self.plan) if not self.layers[p].is_linear), None)
        self.ssm_visit = next((e for e, p in enumerate(self.plan) if self.layers[p].is_linear), None)
        # image input: one shared context; the attention class swap adds no parameters
        self.collect_hidden = None  # parity diagnostics: set to a list to receive the embedding + every visit's output
        rp = args.rope_scaling or {}
        self.mrope = MRoPEContext(rp.get("mrope_section", [11, 11, 10]), rp.get("mrope_interleaved", True))
        for layer in self.layers:
            if not layer.is_linear:
                layer.self_attn.__class__ = MRoPEAttention
                layer.self_attn._mrope_ctx = self.mrope

    def _mrope_tables(self, B: int, L: int, offset: int):
        """cos/sin for positions [offset, offset + L): prompt positions where stored, offset + delta past the prompt."""
        ctx = self.mrope
        T = ctx.position_ids.shape[-1]
        parts = []
        if offset < T:
            parts.append(ctx.position_ids[:, :, offset : min(offset + L, T)])
        if offset + L > T:
            start = max(offset, T)
            tail = mx.arange(start, offset + L, dtype=mx.int32)[None, :] + ctx.delta[:, None]  # [B, n]
            parts.append(mx.broadcast_to(tail[None], (3, *tail.shape)))
        pos = parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=-1)
        if pos.shape[1] != B:
            pos = mx.broadcast_to(pos, (3, B, L))
        rope = self.layers[self.plan[self.fa_visit]].self_attn.rope
        inv, mscale = _rope_inv_freq_and_scale(rope)
        half = inv.shape[0]
        axis = mx.array(_mrope_axis_map(half, ctx.section, ctx.interleaved))
        sel = mx.take(pos.astype(mx.float32), axis, axis=0)  # [half, B, L]
        ang = sel.transpose(1, 2, 0) * inv  # [B, L, half]
        ang = mx.concatenate([ang, ang], axis=-1)[:, None]  # [B, 1, L, rot]
        return mx.cos(ang) * mscale, mx.sin(ang) * mscale

    def __call__(self, inputs: mx.array, cache=None, input_embeddings: Optional[mx.array] = None) -> mx.array:
        h = input_embeddings if input_embeddings is not None else self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.plan)
        if len(cache) != len(self.plan):
            raise ValueError(f"cache has {len(cache)} slots, the execution plan needs {len(self.plan)}")
        fa_mask = create_attention_mask(h, cache[self.fa_visit]) if self.fa_visit is not None else None
        ssm_mask = create_ssm_mask(h, cache[self.ssm_visit]) if self.ssm_visit is not None else None
        if self.mrope.active and self.fa_visit is not None:
            c0 = cache[self.fa_visit]
            self.mrope.cos, self.mrope.sin = self._mrope_tables(h.shape[0], h.shape[1], int(c0.offset) if c0 is not None else 0)
        col = self.collect_hidden
        if col is not None:
            col.append(h)
        try:
            for physical, c in zip(self.plan, cache):
                layer = self.layers[physical]
                h = layer(h, mask=ssm_mask if layer.is_linear else fa_mask, cache=c)
                if col is not None:
                    col.append(h)
        finally:
            self.mrope.cos = self.mrope.sin = None
        return self.norm(h)


class TextModel(q35.TextModel):
    def __init__(self, args: q35.TextModelArgs, loop: dict, mhc: dict):
        nn.Module.__init__(self)
        self.args = args
        self.model_type = args.model_type
        self.model = SenseNovaLoopTextModel(args, loop, mhc)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        self.sharding_group = None

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [ArraysCache(size=2) if self.model.layers[p].is_linear else KVCache() for p in self.model.plan]

    @property
    def cast_predicate(self):
        def predicate(path: str):
            return not (path.endswith("A_log") or ".mhc_attn." in path or ".mhc_ffn." in path)

        return predicate


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.language_model = TextModel(q35.TextModelArgs.from_dict(args.text_config), args.loop_settings, args.mhc)
        _use_cast_norms(self.language_model)

    def __call__(self, inputs: mx.array, cache=None, input_embeddings: Optional[mx.array] = None):
        return self.language_model(inputs, cache=cache, input_embeddings=input_embeddings)

    # ---- image input (the vision tower lives outside this module: sensenova_mlx/vision.py) ----
    def set_mrope(self, position_ids, rope_deltas):
        """Activate 3-axis positions for the prompt about to be prefilled (from get_rope_index) and the decode delta.
        Stays active for the following decode steps; call clear_mrope() before an unrelated prompt."""
        ctx = self.language_model.model.mrope
        ctx.position_ids = mx.array(np.asarray(position_ids), dtype=mx.int32)
        ctx.delta = mx.array(np.asarray(rope_deltas).reshape(-1), dtype=mx.int32)

    def clear_mrope(self):
        ctx = self.language_model.model.mrope
        ctx.position_ids = ctx.delta = ctx.cos = ctx.sin = None

    @property
    def rope_deltas(self):
        return self.language_model.model.mrope.delta

    def embed_with_images(self, input_ids: mx.array, image_features: Optional[mx.array]) -> mx.array:
        """Token embeddings with the merged vision features scattered over <|image_pad|> positions, in order
        (HF masked_scatter)."""
        h = self.language_model.model.embed_tokens(input_ids)
        if image_features is None:
            return h
        mask = input_ids == self.args.image_token_id
        n = int(mask.sum().item())
        if n != image_features.shape[0]:
            raise ValueError(f"{n} image tokens in the prompt, {image_features.shape[0]} image features")
        idx = mx.maximum(mx.cumsum(mask.reshape(-1).astype(mx.int32)) - 1, 0).reshape(mask.shape)
        feats = image_features.astype(h.dtype)[idx]
        return mx.where(mask[..., None], feats, h)

    def prepare_image_prompt(self, input_ids, image_features: Optional[mx.array], image_grid_thw,
                             spatial_merge_size: int = 2) -> mx.array:
        """input_ids [B, T] (+ features from the vision tower) -> input_embeddings; also activates mRoPE positions.
        A prompt without image tokens clears mRoPE and returns plain embeddings (text path unchanged)."""
        ids_np = np.asarray(input_ids)
        if ids_np.ndim == 1:
            ids_np = ids_np[None]
        ids = mx.array(ids_np)
        if image_features is None or not (ids_np == self.args.image_token_id).any():
            self.clear_mrope()
            return self.language_model.model.embed_tokens(ids)
        pos, deltas = get_rope_index(ids_np, image_grid_thw, self.args.image_token_id, spatial_merge_size,
                                     self.args.video_token_id)
        self.set_mrope(pos, deltas)
        return self.embed_with_images(ids, image_features)

    @property
    def model(self):
        return self.language_model.model

    def sanitize(self, weights):
        # q35: drops model.visual.* and mtp.*, renames model.language_model -> language_model.model, conv1d layout,
        # +1 on norms for raw HF checkpoints. Routers stay fp32 (the reference refuses to run them in bf16).
        # The release checkpoint also carries `loop_gate.logits`, which the reference forward never reads: drop it.
        weights = {k: v for k, v in weights.items() if "loop_gate" not in k}
        # optional float32 norm weights (1 + w) shipped as `fp32_norm.<final name>`; they replace the bf16 copies
        fp32_norms = {k[len("fp32_norm."):]: v.astype(mx.float32) for k, v in weights.items() if k.startswith("fp32_norm.")}
        weights = {k: v for k, v in weights.items() if not k.startswith("fp32_norm.")}
        weights = q35.Model.sanitize(self, weights)
        weights.update(fp32_norms)
        return {k: (v.astype(mx.float32) if (".mhc_attn." in k or ".mhc_ffn." in k) else v) for k, v in weights.items()}

    @property
    def layers(self):
        return self.language_model.model.layers

    def make_cache(self):
        return self.language_model.make_cache()

    @property
    def quant_predicate(self):
        return None

    @property
    def cast_predicate(self):
        return self.language_model.cast_predicate
