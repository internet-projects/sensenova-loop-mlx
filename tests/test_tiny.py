#!/usr/bin/env python3
"""Tiny-model checks for the MLX port (CPU, float32). Usage: test_tiny.py <tiny dir> [<tiny no-loop dir>]

1. parity: MLX logits vs the reference logits saved by make_tiny.py (max abs diff, top-1 agreement)
2. cache: prefill + token-by-token decode with the per-visit cache equals the full forward
3. structure: cache slots == plan length (12), routers are float32 after load
4. sinkhorn: columns sum to 1, rows close to 1
5. loop fires: logits differ from the same weights run with eval_count 1 (needs the no-loop dir)
"""
import json, os, sys
from pathlib import Path
import numpy as np
import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sensenova_mlx import sensenova_loop as S
from mlx_lm.utils import load_model

mx.set_default_device(mx.cpu)
mx.set_cache_limit(2 * 2**30)
mx.set_memory_limit(8 * 2**30)
TOL = float(os.environ.get("TINY_TOL", "1e-4"))
fails = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        fails.append(name)


def load(d):
    m, _ = load_model(Path(d), get_model_classes=lambda config: (S.Model, S.ModelArgs))
    return m


d = sys.argv[1]
ref = np.load(os.path.join(d, "ref_logits.npz"))
m = load(d)
ids = mx.array(ref["ids"])
out = np.array(m(ids).astype(mx.float32))
diff = float(np.abs(out - ref["logits"]).max())
top = float((out.argmax(-1) == ref["logits"].argmax(-1)).mean())
check("parity vs reference", diff < TOL and top == 1.0, f"max|dlogit|={diff:.2e} top1={top:.3f}")

x = ids[:1]
full = m(x)
cache = m.make_cache()
check("cache slots == plan", len(cache) == len(m.model.plan) == 12, f"{len(cache)} slots, plan {m.model.plan}")
k = 7
steps = [m(x[:, :k], cache=cache)[:, -1]]
for t in range(k, x.shape[1]):
    steps.append(m(x[:, t : t + 1], cache=cache)[:, -1])
inc = mx.stack(steps, axis=1)
cd = float(mx.abs(inc - full[:, k - 1 :]).max())
check("incremental decode == full forward", cd < TOL, f"max diff {cd:.2e}")

r = m.model.layers[2].mhc_attn
check("routers float32", r.weight.dtype == mx.float32 and r.bias.dtype == mx.float32, str(r.weight.dtype))
sk = S.sinkhorn(mx.random.normal((64, 2, 2)), 20)
check("sinkhorn columns/rows", float(mx.abs(sk.sum(-2) - 1).max()) < 1e-5 and float(mx.abs(sk.sum(-1) - 1).max()) < 1e-3)

if len(sys.argv) > 2:
    m1 = load(sys.argv[2])
    check("no-loop plan", len(m1.model.plan) == 8)
    ld = float(mx.abs(m1(ids) - m(ids)).max())
    check("loop changes the output", ld > 1e-3, f"max diff {ld:.2e}")

print("ALL PASS" if not fails else f"FAILED: {fails}")
sys.exit(1 if fails else 0)
