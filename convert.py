#!/usr/bin/env python3
"""Convert the pinned sensenova/SenseNova-Loop-9B-Preview folder to MLX (text model; vision comes later).

    python convert.py --hf ~/local-llms/sensenova/hf-9937e91 --out ~/local-llms/sensenova/mlx-bf16
    python convert.py --hf ... --out ~/local-llms/sensenova/mlx-q8 --bits 8
    python convert.py --hf ... --out ~/local-llms/sensenova/mlx-q4 --bits 4

Quantization: MLX affine, group size 64, on every Linear and the token embedding (mlx_lm default). Kept as they are:
the 48 mHC routers (float32, never quantized or cast), norms, Gated DeltaNet conv1d / A_log / dt_bias.
The output loads with stock mlx-lm: config.json gets "model_file": "sensenova_loop.py" (trust_remote_code=True), or
with an mlx-lm that ships mlx_lm/models/sensenova_loop.py.
"""
import argparse, importlib, json, os, shutil, sys
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
COPY = ["LICENSE", "chat_template.jinja", "generation_config.json", "preprocessor_config.json", "processor_config.json",
        "video_preprocessor_config.json", ".sensenova_revision", ".sensenova_source"]


def register():
    """Make `mlx_lm.models.sensenova_loop` resolve to this kit's file (no change to the installed mlx-lm)."""
    import mlx_lm.models  # noqa: F401
    mod = importlib.import_module("sensenova_mlx.sensenova_loop")
    sys.modules["mlx_lm.models.sensenova_loop"] = mod
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bits", type=int, default=None, choices=[4, 5, 6, 8])
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--hi", default="", help="comma list of module-path substrings kept at --hi-bits "
                    "(e.g. embed_tokens,lm_head,self_attn,in_proj_a,in_proj_b)")
    ap.add_argument("--hi-bits", type=int, default=8)
    a = ap.parse_args()
    if os.path.exists(a.out):
        sys.exit(f"{a.out} exists; remove it first")
    import mlx.core as mx
    mx.set_cache_limit(int(float(os.environ.get("SN_CACHE_LIMIT_GB", "2")) * 2**30))
    mx.set_memory_limit(int(float(os.environ.get("SN_MEMORY_LIMIT_GB", "48")) * 2**30))
    register()
    from mlx_lm.convert import convert
    hi = [x.strip() for x in a.hi.split(",") if x.strip()]

    def predicate(path, module):
        if any(h in path for h in hi):
            return {"group_size": a.group_size, "bits": a.hi_bits, "mode": "affine"}
        return True

    convert(a.hf, a.out, quantize=a.bits is not None, q_group_size=a.group_size, q_bits=a.bits, dtype=a.dtype,
            quant_predicate=predicate if (hi and a.bits is not None) else None)
    shutil.copy(os.path.join(HERE, "sensenova_mlx", "sensenova_loop.py"), a.out)
    for f in COPY:
        if os.path.exists(os.path.join(a.hf, f)):
            shutil.copy(os.path.join(a.hf, f), a.out)
    cp = os.path.join(a.out, "config.json")
    c = json.load(open(cp))
    c["model_file"] = "sensenova_loop.py"
    # auto_map points at SenseNova's PyTorch modeling file, which an MLX build does not ship; leaving it makes
    # transformers' tokenizer loader ask "run custom code? [y/N]" inside mlx_lm.load.
    c.pop("auto_map", None)
    rev = os.path.join(a.hf, ".sensenova_revision")
    c["sensenova_mlx"] = {"source": "sensenova/SenseNova-Loop-9B-Preview",
                          "revision": open(rev).read().strip() if os.path.exists(rev) else None,
                          "text_only": True, "hi_bits_for": hi or None}
    json.dump(c, open(cp, "w"), indent=2)
    import mlx.core as mx
    w = {}
    for f in Path(a.out).glob("model*.safetensors"):
        w.update(mx.load(str(f)))
    mk = [k for k in w if ".mhc_" in k]
    bad = [k for k in mk if w[k].dtype != mx.float32 or k.endswith("scales")]
    print(f"mHC tensors: {len(mk)}, not float32 / quantized: {len(bad)}")
    if bad:
        sys.exit("STOP: mHC routers were cast or quantized")


if __name__ == "__main__":
    main()
