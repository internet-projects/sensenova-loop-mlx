#!/usr/bin/env python3
"""MTP head parity, MLX side: backbone hidden states (bf16 build) for a few prompts, then our MLX head on them.
    mtp_parity_mlx.py --model BUILD --mtp mtp.safetensors --out mlx.npz"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--mtp", required=True)
ap.add_argument("--out", required=True); a = ap.parse_args()
import numpy as np, mlx.core as mx
mx.set_cache_limit(2 << 30); mx.set_memory_limit(int(os.environ.get("SN_MEMORY_LIMIT_GB", "30")) << 30)
from mlx_lm import load
from mlx_lm.models.cache import KVCache
from sensenova_mlx.mtp import load_mtp_head
model, tok = load(a.model, trust_remote_code=True)
head = load_mtp_head(model, a.mtp)
lm = model.language_model
prompts = ["Explain why the sky is blue in two sentences.",
           "Write a Python function that returns the n-th Fibonacci number.",
           "The capital of France is Paris. The capital of Italy is Rome. The capital of Greece is",
           "List three facts about the Moon, one per line."]
out = {}
for i, p in enumerate(prompts):
    text = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False, enable_thinking=False)
    ids = tok.encode(text)
    x = mx.array(ids)[None]
    h = lm.model(x, cache=model.make_cache())               # post-norm final hidden (B, P, D)
    emb = lm.model.embed_tokens(x)
    c = KVCache()
    mh = head(h[:, :-1], emb[:, 1:], c)                     # positions 1..P-1
    lg = lm.lm_head(mh)
    mx.eval(h, emb, mh, lg)
    out[f"ids{i}"] = np.array(ids, dtype=np.int64)
    out[f"h{i}"] = np.array(h.astype(mx.float32)[0])
    out[f"emb{i}"] = np.array(emb.astype(mx.float32)[0])
    out[f"mh{i}"] = np.array(mh.astype(mx.float32)[0])
    out[f"top{i}"] = np.array(mx.argmax(lg, axis=-1)[0])
    out[f"tgt_next2_{i}"] = np.array(ids[2:], dtype=np.int64)  # actual token t+2 for each MTP position (teacher forced)
    print(f"prompt {i}: {len(ids)} tokens", flush=True)
np.savez(a.out, **out)
print("saved", a.out)
