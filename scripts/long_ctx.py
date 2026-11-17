#!/usr/bin/env python3
"""Long-context needle check for an MLX build (aggregate pass/fail only).

  long_ctx.py --model BUILD --tokens 65536 [--tokens 131072 ...] --out res.json

Filler text with a random 6-digit pass code hidden at 10%, 50% and 90% depth; chat template, thinking off, greedy,
24 new tokens; pass when the reply contains the code. Uses mlx-lm's chunked prefill (2048) and the per-visit cache.
"""
import argparse, json, os, random, time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokens", type=int, action="append", required=True)
    ap.add_argument("--depths", default="0.1,0.5,0.9")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import mlx.core as mx
    mx.set_cache_limit(int(float(os.environ.get("SN_CACHE_LIMIT_GB", "2")) * 2**30))
    mx.set_memory_limit(int(float(os.environ.get("SN_MEMORY_LIMIT_GB", "30")) * 2**30))
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler
    model, tok = load(a.model, trust_remote_code=True)
    filler = ("The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. ")
    fl = len(tok.encode(filler))
    rng = random.Random(0)
    res = {"model": os.path.basename(a.model.rstrip("/")), "runs": []}
    for n in a.tokens:
        for d in [float(x) for x in a.depths.split(",")]:
            code = str(rng.randint(100000, 999999))
            reps = max(1, (n - 200) // fl)
            cut = int(reps * d)
            body = filler * cut + f" The secret pass code is {code}. Remember it. " + filler * (reps - cut)
            msgs = [{"role": "user", "content": body + "\n\nWhat is the secret pass code? Reply with the number only."}]
            prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
            ids = tok.encode(prompt)
            t = time.time()
            out = generate(model, tok, prompt=ids, max_tokens=24, sampler=make_sampler(temp=0.0))
            dt = time.time() - t
            ok = code in out
            res["runs"].append({"tokens": len(ids), "depth": d, "pass": ok, "seconds": round(dt, 1)})
            print(f"ctx {len(ids)} depth {d}: {'PASS' if ok else 'FAIL'} ({dt:.0f}s) peak {mx.get_peak_memory()/2**30:.1f} GB", flush=True)
            mx.clear_cache()
    res["peak_gb"] = round(mx.get_peak_memory() / 2**30, 1)
    res["passed"] = sum(r["pass"] for r in res["runs"])
    res["total"] = len(res["runs"])
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"LONGCTX {res['passed']}/{res['total']} peak {res['peak_gb']} GB")


if __name__ == "__main__":
    main()
