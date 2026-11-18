#!/usr/bin/env python3
"""Decode speed: plain greedy vs MTP speculative greedy (k = 1, 2, 3) on one build; 3 prompts, 256 new tokens each,
thinking off. Prefill is excluded: tok/s counts generated tokens over the decode time only.
    mtp_speed.py --model BUILD --mtp mtp.safetensors"""
import argparse, os, statistics, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--mtp", required=True)
ap.add_argument("--gen", type=int, default=256); a = ap.parse_args()
import mlx.core as mx
mx.set_cache_limit(2 << 30); mx.set_memory_limit(24 << 30)
from mlx_lm import load
from sensenova_mlx import mtp as M
model, tok = load(a.model, trust_remote_code=True)
head = M.load_mtp_head(model, a.mtp)
lm = model.language_model
P = ["Write a detailed explanation of how a refrigerator works.", "Write a Python class for a bank account with deposit, withdraw and a transaction history.",
     "Tell a story about a lighthouse keeper who finds a message in a bottle."]


def ids_of(p):
    return tok.encode(tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False, enable_thinking=False))


def plain(ids, n):
    c = model.make_cache(); lg = M._logits(lm, lm.model(mx.array(ids)[None], cache=c)); t = int(mx.argmax(lg[0, -1]).item())
    t0 = time.perf_counter(); k = 1
    while k < n:
        lg = M._logits(lm, lm.model(mx.array([[t]]), cache=c)); t = int(mx.argmax(lg[0, -1]).item()); k += 1
    return (n - 1) / (time.perf_counter() - t0)


def spec(ids, n, k):
    st = M.MTPStats(); g = M.generate_mtp(model, head, ids, max_tokens=n, k=k, stats=st)
    next(g); t0 = time.perf_counter(); m = 1
    for _ in g:
        m += 1
    return (m - 1) / (time.perf_counter() - t0), st.acceptance


plain(ids_of("Hello"), 16); list(M.generate_mtp(model, head, ids_of("Hello"), max_tokens=16, k=2))  # warm-up
res = {"plain": []}
for p in P:
    ids = ids_of(p)
    res["plain"].append(plain(ids, a.gen))
    for k in (1, 2, 3):
        tps, acc = spec(ids, a.gen, k)
        res.setdefault(f"k{k}", []).append(tps); res.setdefault(f"acc{k}", []).append(acc)
base = statistics.median(res["plain"])
print(f"SPEED plain greedy: {base:.1f} tok/s (median of {len(P)})", flush=True)
for k in (1, 2, 3):
    s = statistics.median(res[f"k{k}"])
    print(f"SPEED MTP k={k}: {s:.1f} tok/s ({s / base:.2f}x) | acceptance {statistics.mean(res[f'acc{k}']):.2f}", flush=True)
print(f"peak memory {mx.get_peak_memory() / 2**30:.1f} GB | fp32 logits {M.FP32_LOGITS[0]}", flush=True)
