#!/usr/bin/env python3
"""MTP speculative decoding tests on one build.
    mtp_test.py --model BUILD --mtp mtp.safetensors --mode greedy|sample|long [--k 3] [--n 16] [--max 128]

greedy: token-identical to plain greedy decode on N prompts (thinking off), plus acceptance stats.
sample: temperature 0.7 sanity (coherent output, acceptance, mean log-prob under the target vs plain sampling).
long:   one ~8K-token prompt, greedy identity on 64 tokens.
"""
import argparse, os, statistics, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--mtp", required=True)
ap.add_argument("--mode", default="greedy"); ap.add_argument("--k", type=int, default=3)
ap.add_argument("--n", type=int, default=16); ap.add_argument("--idx", default=""); ap.add_argument("--max", type=int, default=128)
a = ap.parse_args()
import mlx.core as mx
mx.set_cache_limit(2 << 30); mx.set_memory_limit(int(os.environ.get("SN_MEMORY_LIMIT_GB", "24")) << 30)
from mlx_lm import load
from sensenova_mlx.mtp import MTPStats, generate_mtp, load_mtp_head
from sensenova_mlx import mtp as M
if os.environ.get("SN_FP32_LOGITS", "1") == "0":
    M.FP32_LOGITS[0] = False

model, tok = load(a.model, trust_remote_code=True)
head = load_mtp_head(model, a.mtp)
lm = model.language_model
EOS = set(tok.eos_token_ids) if hasattr(tok, "eos_token_ids") else {tok.eos_token_id}

PROMPTS = [
    "Explain why the sky is blue in three sentences.", "Write a Python function that checks if a string is a palindrome.",
    "What are the main differences between TCP and UDP?", "Summarize the plot of Romeo and Juliet in one paragraph.",
    "Give me five tips for writing clear emails.", "Translate to French: 'The weather is nice today and we are going to the beach.'",
    "What is 37 times 43? Show the steps.", "Write a haiku about autumn leaves.",
    "Explain recursion to a ten-year-old.", "List the planets of the solar system in order from the Sun.",
    "Write a SQL query that returns the ten most recent orders per customer.", "What causes the seasons on Earth?",
    "Describe how a bicycle gear system works.", "Write a short bedtime story about a fox and a lantern.",
    "What is the difference between a list and a tuple in Python?", "Give three arguments for and against remote work.",
]


def chat_ids(p, think=False):
    return tok.encode(tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True,
                                              tokenize=False, enable_thinking=think))


MARGINS = []


def plain_greedy(ids, n):
    MARGINS.clear()
    cache = model.make_cache()
    x = mx.array(ids)[None]
    for s in range(0, len(ids), 2048):
        logits = M._logits(lm, lm.model(x[:, s:s + 2048], cache=cache))
    out = []
    def top(l):
        l = l.astype(mx.float32); s = mx.sort(l)[-2:]; MARGINS.append(float((s[1] - s[0]).item())); return int(mx.argmax(l).item())
    t = top(logits[0, -1])
    while True:
        out.append(t)
        if t in EOS or len(out) >= n:
            return out
        logits = M._logits(lm, lm.model(mx.array([[t]]), cache=cache))
        t = top(logits[0, -1])


def mtp_tokens(ids, n, k, temp=0.0, seed=None):
    st = MTPStats()
    out = list(generate_mtp(model, head, ids, max_tokens=n, k=k, temperature=temp, eos=EOS, seed=seed, stats=st))
    return out, st


if a.mode == "greedy":
    same = 0; acc = []; first_div = []
    sel = [int(x) for x in a.idx.split(",")] if a.idx else list(range(a.n))
    for i in sel:
        p = PROMPTS[i]
        ids = chat_ids(p)
        ref = plain_greedy(ids, a.max)
        got, st = mtp_tokens(ids, a.max, a.k)
        ok = got == ref
        same += ok; acc.append(st.acceptance)
        if not ok:
            d = next((j for j, (x, y) in enumerate(zip(got, ref)) if x != y), min(len(got), len(ref)))
            first_div.append(d)
            kind = st.trace[d] if d < len(st.trace) else "?"
            prev = st.trace[d - 1] if d > 0 else "-"
            margin = MARGINS[d] if d < len(MARGINS) else float("nan")
        print(f"greedy k={a.k} | prompt {i:2d} | identical {ok} | tokens {len(got)}/{len(ref)} | acceptance {st.acceptance:.2f} "
              f"| rounds {st.rounds}" + ("" if ok else f" | first diff at {first_div[-1]} ({kind}, after {prev}) | plain top-2 margin {margin:.3f}"), flush=True)
    print(f"MTP-GREEDY k={a.k}: identical {same}/{len(acc)} | mean acceptance {statistics.mean(acc):.3f} "
          f"| tokens per target call ~{1 + a.k * statistics.mean(acc):.2f}", flush=True)
elif a.mode == "sample":
    acc = []
    for i, p in enumerate(PROMPTS[:4]):
        ids = chat_ids(p)
        got, st = mtp_tokens(ids, a.max, a.k, temp=0.7, seed=1234 + i)
        # mean log-prob of the produced continuation under the target (teacher-forced), vs plain sampled continuation
        def lp(cont):
            x = mx.array(ids + cont)[None]
            lg = lm(x, cache=model.make_cache())[0, len(ids) - 1:-1].astype(mx.float32)
            l = lg / 0.7 - mx.logsumexp(lg / 0.7, axis=-1, keepdims=True)
            return float(mx.mean(mx.take_along_axis(l, mx.array(cont)[:, None], axis=-1)).item())
        mx.random.seed(99 + i)
        cache = model.make_cache(); plain = []; x = mx.array(ids)[None]
        logits = lm(x, cache=cache)
        while len(plain) < a.max:
            t = int(mx.random.categorical(logits[0, -1].astype(mx.float32) / 0.7).item()); plain.append(t)
            if t in EOS: break
            logits = lm(mx.array([[t]]), cache=cache)
        acc.append(st.acceptance)
        print(f"sample k={a.k} | prompt {i} | acceptance {st.acceptance:.2f} | mean logp mtp {lp(got):.3f} plain {lp(plain):.3f} "
              f"| {tok.decode(got)[:90]!r}", flush=True)
    print(f"MTP-SAMPLE k={a.k}: mean acceptance {statistics.mean(acc):.3f}", flush=True)
elif a.mode == "long":
    filler = "The quick brown fox jumps over the lazy dog near the quiet river bank. "
    p = filler * 560 + "\nSummarize the text above in one sentence."
    ids = chat_ids(p)
    t0 = time.time(); ref = plain_greedy(ids, 64); t1 = time.time()
    got, st = mtp_tokens(ids, 64, a.k); t2 = time.time()
    print(f"MTP-LONG k={a.k}: prompt {len(ids)} tok | identical {got == ref} | acceptance {st.acceptance:.2f} | "
          f"{tok.decode(got)[:80]!r}", flush=True)
