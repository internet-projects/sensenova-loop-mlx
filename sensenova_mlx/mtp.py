# Copyright © 2026 NetProjects (Paris Michalas). Apache-2.0.
# MTP speculative decoding for the SenseNova-Loop MLX port.
#
# The MTP head (sensenova/SenseNova-Loop-9B `mtp.safetensors`, PR #12) is one Qwen3.5 full-attention decoder layer
# that reads the backbone's final (post-norm) hidden state of token t and the embedding of token t+1, and predicts
# token t+2 through the shared `lm_head`. Same algorithm as SenseNova's `generate_mtp` (OpenSenseNova/SenseNova-Loop
# PR #4, Apache-2.0): draft `k` tokens with the head, verify the block with one target forward, accept the matching
# prefix (greedy) or use ratio acceptance with a residual draw (sampling).
#
# Difference from SenseNova's reference: no replay forward on rejection. Full-attention visits are trimmed, and every
# Gated DeltaNet visit slot records its recurrence inputs during verify, so its state is rebuilt exactly for the
# accepted prefix (same idea as our DSpine hybrid rollback, one record per *visit slot* because the loop reuses the
# layer modules). The hidden states of the accepted prefix come from the verify pass itself (causal).
# Not affiliated with SenseTime / SenseNova.

import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn

try:
    from mlx_lm.models import qwen3_5 as q35
    from mlx_lm.models.base import create_attention_mask, scaled_dot_product_attention
    from mlx_lm.models.cache import ArraysCache, KVCache
    from mlx_lm.models.gated_delta import gated_delta_update, normalize_qk
except ImportError as e:  # pragma: no cover
    raise ImportError("mlx-lm >= 0.32 is required") from e


# ----------------------------------------------------------------------------------------------------------- head
class _ShiftedAttention(q35.Attention):
    """Qwen3.5 attention whose rope position = cache.offset + shift (the MTP cache entry j sits at position j+shift)."""

    shift: int = 1

    def __call__(self, x, mask=None, cache=None):
        B, L, D = x.shape
        q_out = self.q_proj(x)
        queries, gate = mx.split(q_out.reshape(B, L, self.num_attention_heads, -1), 2, axis=-1)
        gate = gate.reshape(B, L, -1)
        keys, values = self.k_proj(x), self.v_proj(x)
        queries = self.q_norm(queries).transpose(0, 2, 1, 3)
        keys = self.k_norm(keys.reshape(B, L, self.num_key_value_heads, -1)).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)
        off = cache.offset + self.shift
        queries = self.rope(queries, offset=off)
        keys = self.rope(keys, offset=off)
        keys, values = cache.update_and_fetch(keys, values)
        out = scaled_dot_product_attention(queries, keys, values, cache=cache, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(out * mx.sigmoid(gate))


class MTPHead(nn.Module):
    """`mtp.*` of the checkpoint: norm_e(embed) ++ norm_h(hidden) -> fc -> one full-attention layer -> norm."""

    def __init__(self, args: "q35.TextModelArgs"):
        super().__init__()
        self.pre_fc_norm_embedding = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.pre_fc_norm_hidden = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.fc = nn.Linear(2 * args.hidden_size, args.hidden_size, bias=False)
        layer = q35.DecoderLayer(args, args.full_attention_interval - 1)  # a full-attention layer
        assert not layer.is_linear
        layer.self_attn.__class__ = _ShiftedAttention
        self.layers = [layer]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, hidden: mx.array, embeds: mx.array, cache: KVCache) -> mx.array:
        x = self.fc(mx.concatenate([self.pre_fc_norm_embedding(embeds), self.pre_fc_norm_hidden(hidden)], axis=-1))
        mask = create_attention_mask(x, cache) if x.shape[1] > 1 else None
        x = self.layers[0](x, mask=mask, cache=cache)
        return self.norm(x)


_NORM_SUFFIXES = (".input_layernorm.weight", ".post_attention_layernorm.weight", ".q_norm.weight", ".k_norm.weight",
                  "pre_fc_norm_embedding.weight", "pre_fc_norm_hidden.weight", "mtp.norm.weight")


def sanitize_mtp(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
    """Raw HF `mtp.*` tensors -> MTPHead parameter names. Qwen3.5 RMSNorm stores (w - 1): add 1 back, as mlx-lm's
    qwen3_5 sanitize does for the backbone."""
    out = {}
    for k, v in weights.items():
        if not k.startswith("mtp."):
            continue
        if v.ndim == 1 and any(k.endswith(s) for s in _NORM_SUFFIXES):
            v = v.astype(mx.float32) + 1.0
        out[k[len("mtp."):]] = v
    return out


def load_mtp_head(model, mtp_file: str, dtype=mx.bfloat16, bits: Optional[int] = None, group_size: int = 64) -> MTPHead:
    """Build the head for `model` (our SenseNova-Loop MLX Model) from a standalone `mtp.safetensors`."""
    lm = model.language_model
    head = MTPHead(lm.args)
    w = sanitize_mtp(mx.load(mtp_file))
    w = {k: (v.astype(dtype) if mx.issubdtype(v.dtype, mx.floating) else v) for k, v in w.items()}
    head.load_weights(list(w.items()), strict=True)
    if bits:
        nn.quantize(head, group_size=group_size, bits=bits,
                    class_predicate=lambda p, m: isinstance(m, nn.Linear) and m.weight.shape[-1] % group_size == 0)
    mx.eval(head.parameters())
    return head


# --------------------------------------------------------------------------------------- Gated DeltaNet rollback
class _Record:
    __slots__ = ("conv_input", "ssm0", "q", "k", "v", "a", "b", "steps")

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


_RECORDING = [False]


class RecordingGatedDeltaNet(q35.GatedDeltaNet):
    """mlx-lm 0.32.2 qwen3_5.GatedDeltaNet.__call__ (same compute) that, while recording is on, stores the
    recurrence inputs on the *cache slot* it was called with."""

    def __call__(self, inputs, mask=None, cache=None):
        B, S, _ = inputs.shape
        qkv = self.in_proj_qkv(inputs)
        z = self.in_proj_z(inputs).reshape(B, S, self.num_v_heads, self.head_v_dim)
        b = self.in_proj_b(inputs)
        a = self.in_proj_a(inputs)
        if cache is not None and cache[0] is not None:
            conv_state = cache[0]
        else:
            conv_state = mx.zeros((B, self.conv_kernel_size - 1, self.conv_dim), dtype=inputs.dtype)
        if mask is not None:
            qkv = mx.where(mask[..., None], qkv, 0)
        conv_input = mx.concatenate([conv_state, qkv], axis=1)
        if cache is not None:
            n_keep = self.conv_kernel_size - 1
            if cache.lengths is not None:
                ends = mx.clip(cache.lengths, 0, S)
                positions = (ends[:, None] + mx.arange(n_keep))[..., None]
                cache[0] = mx.take_along_axis(conv_input, positions, axis=1)
            else:
                cache[0] = mx.contiguous(conv_input[:, -n_keep:, :])
        conv_out = nn.silu(self.conv1d(conv_input))
        q, k, v = [
            t.reshape(B, S, h, d)
            for t, h, d in zip(
                mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
                [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                [self.head_k_dim, self.head_k_dim, self.head_v_dim],
            )
        ]
        state = cache[1] if cache else None
        q, k = normalize_qk(q, k, inv_scale=self.head_k_dim**-0.5, eps=1e-6)
        if _RECORDING[0] and cache is not None:
            if mask is not None or cache.lengths is not None:
                raise NotImplementedError("rollback supports unpadded batch-1 decode")
            cache._sn_rec = _Record(conv_input=conv_input, ssm0=state, q=q, k=k, v=v, a=a, b=b, steps=S)
        out, state = gated_delta_update(q, k, v, a, b, self.A_log, self.dt_bias, state, mask,
                                        use_kernel=not self.training)
        if cache is not None:
            cache[1] = state
            cache.advance(S)
        out = self.norm(out, z)
        return self.out_proj(out.reshape(B, S, -1))

    def rollback(self, cache: ArraysCache, keep: int) -> None:
        """Set `cache` to the state after the first `keep` of the recorded steps."""
        rec = getattr(cache, "_sn_rec", None)
        if rec is None:
            raise RuntimeError("no recorded verify pass")
        steps = int(rec.steps)
        if not 1 <= keep <= steps:
            raise ValueError(f"keep={keep} outside [1, {steps}]")
        if keep < steps:
            n_keep = self.conv_kernel_size - 1
            cache[0] = mx.contiguous(rec.conv_input[:, keep: keep + n_keep, :])
            sl = slice(0, keep)
            _, state = gated_delta_update(rec.q[:, sl], rec.k[:, sl], rec.v[:, sl], rec.a[:, sl], rec.b[:, sl],
                                          self.A_log, self.dt_bias, rec.ssm0, None, use_kernel=not self.training)
            cache[1] = state
            if hasattr(cache, "offset"):
                pass
        cache._sn_rec = None


def install_recording(model) -> None:
    for layer in model.language_model.model.layers:
        if getattr(layer, "is_linear", False) and type(layer.linear_attn) is q35.GatedDeltaNet:
            layer.linear_attn.__class__ = RecordingGatedDeltaNet


def _rollback_target(model, cache: list, keep: int, block: int) -> None:
    """After a verify pass over `block` tokens, keep only the first `keep` in every visit slot."""
    drop = block - keep
    lm = model.language_model.model
    for physical, c in zip(lm.plan, cache):
        layer = lm.layers[physical]
        if layer.is_linear:
            layer.linear_attn.rollback(c, keep)
        elif drop:
            c.trim(drop)


# ----------------------------------------------------------------------------------------------- generation
@dataclass
class MTPStats:
    draft_tokens: int = 0
    accepted_tokens: int = 0
    rounds: int = 0
    target_calls: int = 0
    trace: list = field(default_factory=list)  # per yielded token: 'first' | 'acc' | 'corr' | 'bonus'

    @property
    def acceptance(self) -> float:
        return self.accepted_tokens / self.draft_tokens if self.draft_tokens else 0.0


FP32_LOGITS = [True]  # compute the lm_head in float32 (fewer exact bf16 ties in greedy argmax)


def _logits(lm, h):
    return lm.lm_head(h.astype(mx.float32) if FP32_LOGITS[0] else h)


def _target(model, tokens: mx.array, cache: list):
    """One target forward -> (post-norm hidden, logits) for every position."""
    lm = model.language_model
    h = lm.model(tokens, cache=cache)
    return h, _logits(lm, h)


def generate_mtp(model, head: MTPHead, prompt: List[int], *, max_tokens: int = 256, k: int = 3,
                 temperature: float = 0.0, top_p: float = 1.0, eos: Optional[set] = None,
                 seed: Optional[int] = None, prefill_step: int = 2048, stats: Optional[MTPStats] = None):
    """Speculative decoding with the MTP head. Yields token ids one by one (accepted + correction tokens).

    temperature == 0: greedy; output is token-identical to plain greedy decoding of the target.
    temperature > 0: ratio acceptance with residual correction (target distribution preserved).
    """
    install_recording(model)
    lm = model.language_model
    embed = lm.model.embed_tokens
    eos = set(eos or ())
    stats = stats if stats is not None else MTPStats()
    if seed is not None:
        mx.random.seed(seed)
    greedy = temperature <= 0

    def probs(logits):  # logits (V,) float32 -> distribution
        logits = logits.astype(mx.float32) / temperature
        p = mx.softmax(logits, axis=-1)
        if top_p < 1.0:
            sp = mx.sort(p)[::-1]
            cs = mx.cumsum(sp)
            cutoff = sp[mx.minimum(mx.sum(cs < top_p), sp.shape[0] - 1)]
            p = mx.where(p >= cutoff, p, 0)
            p = p / p.sum()
        return p

    def pick(logits):
        if greedy:
            return int(mx.argmax(logits).item()), None
        p = probs(logits)
        return int(mx.random.categorical(mx.log(p + 1e-30)).item()), p

    cache = model.make_cache()
    ids = mx.array(prompt)[None]
    P = len(prompt)
    # prefill in chunks; keep the hidden states for priming the draft cache
    hs = []
    for s in range(0, P, prefill_step):
        h, logits = _target(model, ids[:, s:s + prefill_step], cache)
        mx.eval(h)
        hs.append(h)
        stats.target_calls += 1
    hidden = mx.concatenate(hs, axis=1)
    previous = hidden[:, -1:]
    first, _ = pick(logits[0, -1])
    stats.trace.append("first")
    yield first
    produced = 1
    if first in eos or produced >= max_tokens:
        return
    draft_cache = KVCache()
    if P > 1:  # entry j (position j) pairs the target hidden of token j-1 with the embedding of token j
        for s in range(1, P, prefill_step):
            e = min(P, s + prefill_step)
            _ = head(hidden[:, s - 1:e - 1], embed(ids[:, s:e]), draft_cache)
        mx.eval(draft_cache.keys)
    del hs, hidden
    last = first
    while produced < max_tokens:
        n = min(k, max_tokens - produced)
        draft_len = draft_cache.offset
        # ---- draft
        proposals, qdists = [], []
        dh, tok = previous, last
        for _ in range(n):
            dh = head(dh, embed(mx.array([[tok]])), draft_cache)
            dl = _logits(lm, dh)[0, -1]
            tok, q = pick(dl)
            proposals.append(tok)
            qdists.append(q)
            if tok in eos:
                break
        # ---- verify
        block = [last] + proposals
        _RECORDING[0] = True
        try:
            vh, vlogits = _target(model, mx.array([block]), cache)
        finally:
            _RECORDING[0] = False
        stats.target_calls += 1
        stats.rounds += 1
        stats.draft_tokens += len(proposals)
        accepted = 0
        correction = None
        out_tokens = []
        for i, cand in enumerate(proposals):
            if greedy:
                t = int(mx.argmax(vlogits[0, i]).item())
                if t != cand:
                    correction = t
                    break
            else:
                p = probs(vlogits[0, i])
                q = qdists[i]
                ratio = (p[cand] / mx.maximum(q[cand], 1e-30)).item()
                if mx.random.uniform().item() >= min(1.0, ratio):
                    r = mx.maximum(p - q, 0)
                    r = r / mx.maximum(r.sum(), 1e-30)
                    correction = int(mx.random.categorical(mx.log(r + 1e-30)).item())
                    break
            accepted += 1
            out_tokens.append(cand)
            if cand in eos or produced + len(out_tokens) >= max_tokens:
                break
        stats.accepted_tokens += accepted
        stopped = any(t in eos for t in out_tokens) or produced + len(out_tokens) >= max_tokens
        if correction is None and not stopped:
            correction, _ = pick(vlogits[0, len(proposals)])  # all accepted: bonus token from the last position
        keep = accepted + 1  # `last` + accepted proposals are now part of the target cache
        _rollback_target(model, cache, keep, len(block))
        for t in out_tokens:
            stats.trace.append("acc")
            yield t
        produced += len(out_tokens)
        if stopped:
            return
        stats.trace.append("bonus" if accepted == len(proposals) else "corr")
        yield correction
        produced += 1
        if correction in eos or produced >= max_tokens:
            return
        # ---- rebuild this round's draft entries from the exact target hidden states
        draft_cache.trim(draft_cache.offset - draft_len)
        committed = mx.array([block[:keep]])
        shifted = mx.concatenate([previous, vh[:, : keep - 1]], axis=1)
        _ = head(shifted, embed(committed), draft_cache)
        previous = vh[:, keep - 1: keep]
        last = correction
