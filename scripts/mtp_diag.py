#!/usr/bin/env python3
"""Why MTP greedy differs from plain greedy: block-verify numerics vs a rollback bug.
For prompts where they differ, take the plain greedy continuation R and compare, at every position, the target's
argmax when R is fed token by token (plain decode) vs fed in blocks of 4 (as verify does) vs fed in blocks of 4 after
an MTP-style partial-accept rollback. Reports flips and the top-2 margin at the flips.
    mtp_diag.py --model BUILD --mtp mtp.safetensors"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--mtp", required=True)
a = ap.parse_args()
import mlx.core as mx
mx.set_cache_limit(2 << 30); mx.set_memory_limit(24 << 30)
from mlx_lm import load
from sensenova_mlx import mtp as M
model, tok = load(a.model, trust_remote_code=True)
M.install_recording(model)
lm = model.language_model
P = ["Give me five tips for writing clear emails.", "Write a SQL query that returns the ten most recent orders per customer.",
     "What is the difference between a list and a tuple in Python?", "Describe how a bicycle gear system works."]


def ids_of(p):
    return tok.encode(tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False, enable_thinking=False))


for p in P:
    ids = ids_of(p)
    # plain greedy reference + per-step logits
    c = model.make_cache(); lg = lm(mx.array(ids)[None], cache=c); R = []; step_lg = []
    for _ in range(64):
        l = lg[0, -1].astype(mx.float32); step_lg.append(l); t = int(mx.argmax(l).item()); R.append(t)
        lg = lm(mx.array([[t]]), cache=c)
    # blocks of 4 without rollback
    c = model.make_cache(); lg = lm(mx.array(ids)[None], cache=c); blk = [lg[0, -1].astype(mx.float32)]
    seq = R[:-1]
    for s in range(0, len(seq), 4):
        out = lm(mx.array([seq[s:s + 4]]), cache=c)
        blk += [out[0, i].astype(mx.float32) for i in range(out.shape[1])]
    # blocks of 4 with a fake partial accept: feed 4 tokens where the last 2 are junk, roll back to 2, continue
    c = model.make_cache(); lg = lm(mx.array(ids)[None], cache=c); rb = [lg[0, -1].astype(mx.float32)]
    s = 0
    while s < len(seq):
        good = seq[s:s + 2]
        junk = [11, 13][: 4 - len(good)] if len(good) == 2 else []
        M._RECORDING[0] = True
        out = lm(mx.array([good + junk]), cache=c)
        M._RECORDING[0] = False
        M._rollback_target(model, c, len(good), len(good) + len(junk))
        rb += [out[0, i].astype(mx.float32) for i in range(len(good))]
        s += 2
    n = min(len(step_lg), len(blk), len(rb))
    def flips(other):
        f = []
        for i in range(n):
            a1 = int(mx.argmax(step_lg[i]).item()); a2 = int(mx.argmax(other[i]).item())
            if a1 != a2:
                top = mx.sort(step_lg[i])[-2:]
                f.append((i, round(float((top[1] - top[0]).item()), 3)))
        return f
    d_blk = float(mx.max(mx.abs(mx.stack(step_lg[:n]) - mx.stack(blk[:n]))).item())
    d_rb = float(mx.max(mx.abs(mx.stack(blk[:n]) - mx.stack(rb[:n]))).item())
    print(f"prompt {p[:40]!r}: block-vs-step flips {flips(blk)} | rollback-vs-step flips {flips(rb)} | "
          f"max|dlogit| step-vs-block {d_blk:.3f} block-vs-rollback {d_rb:.3f}", flush=True)
