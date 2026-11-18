#!/usr/bin/env python3
"""Test folders with float32 norm weights: for each build, a folder of symlinks to the published files plus
`model.fp32norms.safetensors` (keys `fp32_norm.<name>` = 1 + w in float32 from the HF checkpoint) and this repo's
sensenova_loop.py. Never writes into the build folders.
    make_normfix.py --hf HF_DIR --build BUILD --out OUT_DIR"""
import argparse, json, os, shutil
ap = argparse.ArgumentParser(); ap.add_argument("--hf", required=True); ap.add_argument("--build", required=True)
ap.add_argument("--out", required=True); a = ap.parse_args()
import mlx.core as mx
idx = json.load(open(os.path.join(a.hf, "model.safetensors.index.json")))["weight_map"]
SUF = (".input_layernorm.weight", ".post_attention_layernorm.weight", ".q_norm.weight", ".k_norm.weight")
names = [k for k in idx if k.startswith("model.language_model.") and (k.endswith(SUF) or k == "model.language_model.norm.weight")]
bidx = json.load(open(os.path.join(a.build, "model.safetensors.index.json")))["weight_map"]
out = {}
shards = {}
for k in names:
    f = idx[k]
    if f not in shards:
        shards[f] = mx.load(os.path.join(a.hf, f))
    w = shards[f][k].astype(mx.float32) + 1.0
    final = k.replace("model.language_model.", "language_model.model.", 1)
    if final not in bidx:
        raise SystemExit(f"{final} not in the build index")
    out["fp32_norm." + final] = w
assert len(out) == 81, len(out)
os.makedirs(a.out, exist_ok=True)
for f in os.listdir(a.build):
    src = os.path.join(a.build, f)
    if f in ("sensenova_loop.py",) or f.startswith(".") or not os.path.isfile(src):
        continue
    dst = os.path.join(a.out, f)
    if not os.path.exists(dst):
        os.symlink(src, dst)
shutil.copy2(os.path.join(os.path.dirname(__file__), "..", "sensenova_mlx", "sensenova_loop.py"), os.path.join(a.out, "sensenova_loop.py"))
mx.save_safetensors(os.path.join(a.out, "model.fp32norms.safetensors"), out)
print(f"{a.out}: {len(out)} fp32 norm tensors, {sum(v.size for v in out.values())*4/1e3:.0f} KB")
