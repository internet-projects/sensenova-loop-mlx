# Copyright © 2026 NetProjects (Paris Michalas). Apache-2.0.
# MLX port of SenseNova-Loop (sensenova/SenseNova-Loop-9B-Preview, modeling_sensenova_loop.py @ 9937e91).
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
    from .base import BaseModelArgs, create_attention_mask, create_ssm_mask
    from .cache import ArraysCache, KVCache
except ImportError:  # loaded as a model_file
    from mlx_lm.models import qwen3_5 as q35
    from mlx_lm.models.base import BaseModelArgs, create_attention_mask, create_ssm_mask
    from mlx_lm.models.cache import ArraysCache, KVCache


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    text_config: dict
    loop_settings: dict = field(default_factory=dict)
    mhc: dict = field(default_factory=dict)


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

    def __call__(self, inputs: mx.array, cache=None, input_embeddings: Optional[mx.array] = None) -> mx.array:
        h = input_embeddings if input_embeddings is not None else self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.plan)
        if len(cache) != len(self.plan):
            raise ValueError(f"cache has {len(cache)} slots, the execution plan needs {len(self.plan)}")
        fa_mask = create_attention_mask(h, cache[self.fa_visit]) if self.fa_visit is not None else None
        ssm_mask = create_ssm_mask(h, cache[self.ssm_visit]) if self.ssm_visit is not None else None
        for physical, c in zip(self.plan, cache):
            layer = self.layers[physical]
            h = layer(h, mask=ssm_mask if layer.is_linear else fa_mask, cache=c)
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

    def __call__(self, inputs: mx.array, cache=None, input_embeddings: Optional[mx.array] = None):
        return self.language_model(inputs, cache=cache, input_embeddings=input_embeddings)

    @property
    def model(self):
        return self.language_model.model

    def sanitize(self, weights):
        # q35: drops model.visual.* and mtp.*, renames model.language_model -> language_model.model, conv1d layout,
        # +1 on norms for raw HF checkpoints. Routers stay fp32 (the reference refuses to run them in bf16).
        weights = q35.Model.sanitize(self, weights)
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
