#!/usr/bin/env python3
"""Short public benchmark for the MLX builds (aggregate numbers only).

  bench.py --model BUILD --data DIR --out res.json [--arc N] [--gsm N]

ARC-Challenge (test): 4-way multiple choice scored by log-likelihood of each answer (acc and length-normalised
acc_norm, lm-eval style "Question: ...\\nAnswer: <choice>"). GSM8K (test, first N): chat template, thinking off,
greedy, up to 384 new tokens, last number in the reply vs the reference answer.
DIR holds arc_challenge_test.parquet and gsm8k_test.parquet (downloaded by the job from allenai/ai2_arc and
openai/gsm8k).
"""
import argparse, json, os, re, sys, time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--arc", type=int, default=0)
    ap.add_argument("--gsm", type=int, default=0)
    a = ap.parse_args()
    import mlx.core as mx
    import numpy as np
    import pyarrow.parquet as pq
    mx.set_cache_limit(int(float(os.environ.get("SN_CACHE_LIMIT_GB", "2")) * 2**30))
    mx.set_memory_limit(int(float(os.environ.get("SN_MEMORY_LIMIT_GB", "24")) * 2**30))
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler
    model, tok = load(a.model, trust_remote_code=True)
    res = {"model": os.path.basename(a.model.rstrip("/"))}
    t0 = time.time()

    def loglik(ctx_ids, cont_ids):
        x = mx.array([ctx_ids + cont_ids])
        lp = model(x[:, :-1])[0].astype(mx.float32)
        lp = lp - mx.logsumexp(lp, axis=-1, keepdims=True)
        tgt = mx.array(cont_ids)
        sel = mx.take_along_axis(lp[len(ctx_ids) - 1:], tgt[:, None], axis=-1)
        v = float(sel.sum())
        mx.clear_cache()
        return v

    if a.arc:
        rows = pq.read_table(Path(a.data) / "arc_challenge_test.parquet").to_pylist()[: a.arc]
        acc = accn = 0
        for r in rows:
            ctx = tok.encode(f"Question: {r['question']}\nAnswer:")
            texts = r["choices"]["text"]; labels = r["choices"]["label"]
            ll = [loglik(ctx, tok.encode(" " + t, add_special_tokens=False)) for t in texts]
            gold = labels.index(r["answerKey"])
            acc += int(int(np.argmax(ll)) == gold)
            accn += int(int(np.argmax([l / max(len(t), 1) for l, t in zip(ll, texts)])) == gold)
        res["arc_challenge"] = {"n": len(rows), "acc": round(100 * acc / len(rows), 2), "acc_norm": round(100 * accn / len(rows), 2)}
        print("ARC-Challenge", res["arc_challenge"], f"{time.time() - t0:.0f}s", flush=True)

    if a.gsm:
        t1 = time.time()
        rows = pq.read_table(Path(a.data) / "gsm8k_test.parquet").to_pylist()[: a.gsm]
        ok = 0; ntok = 0
        sampler = make_sampler(temp=0.0)
        for r in rows:
            msg = [{"role": "user", "content": r["question"] + "\nSolve step by step, then end with: The answer is <number>."}]
            p = tok.apply_chat_template(msg, add_generation_prompt=True, enable_thinking=False)
            out = generate(model, tok, prompt=p, max_tokens=384, sampler=sampler, verbose=False)
            ntok += len(tok.encode(out))
            gold = r["answer"].split("####")[-1].strip().replace(",", "")
            nums = re.findall(r"-?\d[\d,]*\.?\d*", out.replace("$", ""))
            pred = nums[-1].replace(",", "").rstrip(".") if nums else None
            try:
                ok += int(pred is not None and abs(float(pred) - float(gold)) < 1e-6)
            except ValueError:
                pass
            mx.clear_cache()
        res["gsm8k"] = {"n": len(rows), "acc": round(100 * ok / len(rows), 2), "avg_new_tokens": round(ntok / len(rows), 1)}
        print("GSM8K", res["gsm8k"], f"{time.time() - t1:.0f}s", flush=True)
    res["seconds"] = round(time.time() - t0)
    res["peak_gb"] = round(mx.get_peak_memory() / 2**30, 1)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
