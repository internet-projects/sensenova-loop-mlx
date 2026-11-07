#!/usr/bin/env python3
"""Parity between SenseNova's PyTorch remote code and the MLX builds, on a fixed set of short prompts.

  parity.py cases   --hf HF --out cases.json                 # tokenised prompts (fixed list below)
  parity.py ref     --hf HF --cases cases.json --out ref.npz [--device mps|cpu]   # Transformers, bf16
  parity.py mlx     --model BUILD --cases cases.json --out x.npz [--no-loop]
  parity.py compare ref.npz x.npz [--json out.json]

Never run `ref` and `mlx` at the same time: each is its own process and writes to disk.
Stored per prompt: top-20 ids + log-probs at every position, full log-probs at the last position (float16),
and a 32-token greedy continuation. Reported: aggregate numbers only.
"""
import argparse, json, os, sys, time
import numpy as np

PROMPTS = [
    "The capital of Greece is",
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
    "Translate to French: The weather is nice today.\nFrench:",
    "Q: A train leaves at 9:40 and arrives at 13:05. How long is the trip?\nA:",
    "SELECT name, COUNT(*) FROM orders JOIN customers ON",
    "Write a haiku about the sea at night.\n",
    "The three laws of thermodynamics are",
    "In 1969, the Apollo 11 mission",
    "import numpy as np\nx = np.linspace(0, 1, 11)\nprint(",
    "List five prime numbers greater than 100:",
    "The difference between a list and a tuple in Python is",
    "Summarise in one sentence: Photosynthesis converts light energy into chemical energy stored in glucose.",
    "If all bloops are razzies and all razzies are lazzies, then",
    "The chemical formula for table salt is",
    "Dear team,\nPlease find attached",
    "Step-by-step: 17 * 24 =",
]
TOPK, GEN = 20, 32


def log_softmax(x):
    x = x.astype(np.float32)
    m = x.max(-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(-1, keepdims=True))


def pack(logits_per_case, greedy):
    out = {}
    for i, (lg, g) in enumerate(zip(logits_per_case, greedy)):
        lp = log_softmax(lg)
        idx = np.argsort(-lp, -1)[:, :TOPK]
        out[f"{i}_top_ids"] = idx.astype(np.int32)
        out[f"{i}_top_lp"] = np.take_along_axis(lp, idx, -1).astype(np.float32)
        out[f"{i}_last_lp"] = lp[-1].astype(np.float16)
        out[f"{i}_greedy"] = np.array(g, dtype=np.int32)
    return out


def cmd_cases(a):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.hf)
    json.dump({"prompts": PROMPTS, "ids": [tok(p)["input_ids"] for p in PROMPTS], "eos": tok.eos_token_id},
              open(a.out, "w"))


def cmd_ref(a):
    import torch
    from transformers import AutoModelForCausalLM
    c = json.load(open(a.cases))
    t0 = time.time()
    m = AutoModelForCausalLM.from_pretrained(a.hf, trust_remote_code=True, dtype=torch.bfloat16).to(a.device).eval()
    lgs, gr = [], []
    with torch.no_grad():
        for ids in c["ids"]:
            x = torch.tensor([ids], device=a.device)
            lgs.append(m(input_ids=x, use_cache=False).logits[0].float().cpu().numpy())
            g = m.generate(input_ids=x, max_new_tokens=GEN, do_sample=False)[0, len(ids):].tolist()
            gr.append(g)
    np.savez(a.out, **pack(lgs, gr))
    print(f"ref: {len(lgs)} cases in {time.time() - t0:.0f}s")


def cmd_mlx(a):
    import mlx.core as mx
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from convert import register
    register()
    mx.set_cache_limit(int(float(os.environ.get("SN_CACHE_LIMIT_GB", "2")) * 2**30))
    if os.environ.get("SN_MEMORY_LIMIT_GB"):
        mx.set_memory_limit(int(float(os.environ["SN_MEMORY_LIMIT_GB"]) * 2**30))
    from mlx_lm.utils import load_model
    from pathlib import Path
    cfg = {"loop_settings": {**json.load(open(Path(a.model) / "config.json"))["loop_settings"], "eval_count": 1,
                             "counts": [1]}} if a.no_loop else None
    import sensenova_mlx.sensenova_loop as S
    m, _ = load_model(Path(a.model), model_config=cfg, trust_remote_code=True,
                      get_model_classes=lambda config: (S.Model, S.ModelArgs))
    c = json.load(open(a.cases))
    t0 = time.time()
    lgs, gr = [], []
    for ids in c["ids"]:
        x = mx.array([ids])
        lgs.append(np.array(m(x)[0].astype(mx.float32)))
        mx.clear_cache()
        cache = m.make_cache()
        y = m(x, cache=cache)[:, -1]
        g = []
        for _ in range(GEN):
            t = int(mx.argmax(y, -1).item())
            g.append(t)
            if t == c["eos"]:
                break
            y = m(mx.array([[t]]), cache=cache)[:, -1]
        gr.append(g)
        mx.clear_cache()
    np.savez(a.out, **pack(lgs, gr))
    print(f"mlx: {len(lgs)} cases in {time.time() - t0:.0f}s, peak {mx.get_peak_memory() / 2**30:.1f} GB")


def cmd_compare(a):
    r, t = np.load(a.ref), np.load(a.test)
    n = len([k for k in r.files if k.endswith("_greedy")])
    same, tot, dp, kl, gm, gfirst = 0, 0, [], [], 0, []
    for i in range(n):
        ri, ti = r[f"{i}_top_ids"], t[f"{i}_top_ids"]
        same += int((ri[:, 0] == ti[:, 0]).sum()); tot += len(ri)
        # prob of the reference top token under both
        for pos in range(len(ri)):
            tid = ri[pos, 0]
            pt = t[f"{i}_top_lp"][pos][ti[pos] == tid]
            dp.append(abs(np.exp(r[f"{i}_top_lp"][pos, 0]) - (np.exp(pt[0]) if len(pt) else 0.0)))
        lr, lt = r[f"{i}_last_lp"].astype(np.float32), t[f"{i}_last_lp"].astype(np.float32)
        kl.append(float((np.exp(lr) * (lr - lt)).sum()))
        g1, g2 = list(r[f"{i}_greedy"]), list(t[f"{i}_greedy"])
        gm += int(g1 == g2)
        k = next((j for j, (u, v) in enumerate(zip(g1, g2)) if u != v), min(len(g1), len(g2)))
        gfirst.append(k)
    res = {"cases": n, "positions": tot, "same_top1_pct": round(100 * same / tot, 2),
           "top_prob_abs_diff_mean": float(np.mean(dp)), "top_prob_abs_diff_max": float(np.max(dp)),
           "last_kl_mean": float(np.mean(kl)), "last_kl_max": float(np.max(kl)),
           "greedy_exact": gm, "greedy_first_divergence_mean": float(np.mean(gfirst))}
    print(json.dumps(res, indent=1))
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("cases"); p.add_argument("--hf", required=True); p.add_argument("--out", required=True)
    p = sp.add_parser("ref"); p.add_argument("--hf", required=True); p.add_argument("--cases", required=True)
    p.add_argument("--out", required=True); p.add_argument("--device", default="cpu")
    p = sp.add_parser("mlx"); p.add_argument("--model", required=True); p.add_argument("--cases", required=True)
    p.add_argument("--out", required=True); p.add_argument("--no-loop", action="store_true")
    p = sp.add_parser("compare"); p.add_argument("ref"); p.add_argument("test"); p.add_argument("--json")
    a = ap.parse_args()
    {"cases": cmd_cases, "ref": cmd_ref, "mlx": cmd_mlx, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    main()
