"""Independent PyTorch re-implementation of SenseNova-Loop's text forward, on top of Transformers' Qwen3.5 layers.

Used only for the cloud tiny test (the real modeling_sensenova_loop.py was not downloadable from the cloud).
Written from the reference lines in modeling_sensenova_loop.py @ 9937e91; on the Mac, tests/test_tiny.sh also checks
the MLX port against the real remote code.  No cache: full-sequence forward only.
"""
import torch


def execution_plan(n, loop):
    s, e, c = loop["start"], loop["end"], loop["eval_count"]
    return tuple(range(s)) + tuple(range(s, e)) * c + tuple(range(e, n))


def sinkhorn(logits, iterations):
    logits = logits.float()
    for _ in range(iterations):
        logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        logits = logits - torch.logsumexp(logits, dim=-2, keepdim=True)
    return logits.exp()


def route(weight, bias, residual, s, iterations, pre_scale=1.0):
    flat = residual.flatten(-2).float()
    normalized = flat * torch.rsqrt(flat.square().mean(-1, keepdim=True) + 1e-6)
    logits = torch.nn.functional.linear(normalized, weight, bias)
    pre = logits[..., :s].sigmoid()
    post = 2 * logits[..., s:2 * s].sigmoid()
    mix = sinkhorn(logits[..., 2 * s:].unflatten(-1, (s, s)), iterations)
    branch = torch.matmul(pre.to(residual.dtype).unsqueeze(-2), residual).squeeze(-2)
    if pre_scale != 1.0:
        branch = branch * pre_scale
    return branch, post, mix


def merge(residual, branch, post, mix):
    return torch.matmul(mix.to(residual.dtype), residual) + post.to(branch.dtype).unsqueeze(-1) * branch.unsqueeze(-2)


@torch.no_grad()
def forward(text_model, lm_head, mhc_params, input_ids, loop, mhc):
    """text_model: transformers Qwen3_5TextModel; mhc_params[i] = dict(attn=(W, b), ffn=(W, b))."""
    cfg = text_model.config
    h = text_model.embed_tokens(input_ids)
    L = input_ids.shape[1]
    pos = torch.arange(L).view(1, 1, -1).expand(3, input_ids.shape[0], -1)
    pe = text_model.rotary_emb(h, pos)
    text_pos = torch.arange(L).view(1, -1).expand(input_ids.shape[0], -1)
    from transformers.masking_utils import create_causal_mask
    fa_mask = create_causal_mask(config=cfg, inputs_embeds=h, attention_mask=None, past_key_values=None,
                                 position_ids=text_pos)
    s, it = mhc["streams"], mhc["iterations"]
    for p in execution_plan(cfg.num_hidden_layers, loop):
        layer = text_model.layers[p]

        def mixer(x):
            if layer.block_type == "linear_attention":
                return layer.linear_attn(hidden_states=x, cache_params=None, attention_mask=None)
            return layer.self_attn(hidden_states=x, attention_mask=fa_mask, position_ids=text_pos,
                                   past_key_values=None, position_embeddings=pe)[0]

        if mhc["start"] <= p < mhc["end"]:
            residual = h.unsqueeze(-2).expand(*h.shape[:-1], s, h.shape[-1]) if p == mhc["start"] else h
            b, po, mx_ = route(*mhc_params[p]["attn"], residual, s, it)
            residual = merge(residual, mixer(layer.input_layernorm(b)), po, mx_)
            b, po, mx_ = route(*mhc_params[p]["ffn"], residual, s, it)
            residual = merge(residual, layer.mlp(layer.post_attention_layernorm(b)), po, mx_)
            h = residual.mean(-2) if p == mhc["end"] - 1 else residual
        else:
            h = layer(h, position_embeddings=pe, attention_mask=fa_mask if layer.block_type == "full_attention" else None,
                      position_ids=text_pos, past_key_values=None)
    return lm_head(text_model.norm(h))
