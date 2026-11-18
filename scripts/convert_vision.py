#!/usr/bin/env python3
"""Vision weights + a vision-enabled build folder for our MLX SenseNova-Loop-9B builds (local, new folders only).

  # 1. vision tower -> one BF16 file (333 tensors, ~0.91 GB), shared by every text build
  convert_vision.py vision --hf hf --out builds-test/vision.safetensors

  # 2. vision-enabled copy of a text build: weights are symlinked (never modified), small files copied,
  #    sensenova_loop.py / vision.py / processor.py from this kit, config.json gains the vision fields
  convert_vision.py build --src mlx-bf16 \
      --hf hf --vision builds-test/vision.safetensors --out builds-test/mlx-bf16-vl

The text weights are not touched; vision stays BF16 in every variant. Keys in vision.safetensors are
`vision_tower.<module path>` with the Conv3d patch weight [1152, 3, 2, 16, 16] reshaped to Linear [1152, 1536]
(order C, T, ph, pw, which is the processor's per-patch flattening order).
"""
import argparse, hashlib, json, os, shutil, sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

SMALL = ["chat_template.jinja", "generation_config.json", "preprocessor_config.json", "processor_config.json",
         "video_preprocessor_config.json", "tokenizer.json", "tokenizer_config.json", "README.md",
         ".sensenova_revision", ".sensenova_source", "LICENSE"]
VISION_KEYS = ["vision_config", "image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id"]


def cmd_vision(a):
    import mlx.core as mx
    from sensenova_mlx.vision import VisionModel

    mx.set_default_device(mx.cpu)  # 1 GB of tensors; no GPU needed
    mx.set_cache_limit(2 * 2**30)
    mx.set_memory_limit(12 * 2**30)
    if os.path.exists(a.out):
        sys.exit(f"{a.out} exists; remove it first")
    idx = json.load(open(os.path.join(a.hf, "model.safetensors.index.json")))["weight_map"]
    shards = sorted({f for k, f in idx.items() if k.startswith("model.visual.")})
    w = {}
    for s in shards:
        for k, v in mx.load(os.path.join(a.hf, s)).items():
            if k.startswith("model.visual."):
                w[k] = v
    n_expected = sum(k.startswith("model.visual.") for k in idx)
    if len(w) != n_expected:
        sys.exit(f"STOP: read {len(w)} vision tensors, index lists {n_expected}")
    bad = [k for k, v in w.items() if v.dtype != mx.bfloat16]
    if bad:
        sys.exit(f"STOP: non-bf16 vision tensors in the checkpoint: {bad[:5]}")
    cfg = json.load(open(os.path.join(a.hf, "config.json")))["vision_config"]
    vm = VisionModel(cfg)
    sw = VisionModel.sanitize(w)
    vm.load_weights(list(sw.items()), strict=True)  # shape / name check against the MLX module
    out = {f"vision_tower.{k}": v for k, v in sw.items()}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    tmp = a.out + ".partial.safetensors"
    mx.save_safetensors(tmp, out, metadata={"format": "mlx", "source": "model.visual.* of " + os.path.basename(
        os.path.abspath(a.hf)), "dtype": "bfloat16"})
    os.replace(tmp, a.out)
    n = sum(v.size for v in out.values())
    print(f"vision: {len(out)} tensors, {n / 1e6:.1f} M params, {os.path.getsize(a.out) / 1e9:.3f} GB -> {a.out}")


def cmd_build(a):
    src, out = Path(a.src).resolve(), Path(a.out).resolve()
    vision = Path(a.vision).resolve()
    if out.exists():
        sys.exit(f"{out} exists; remove it first")
    if str(out).startswith(str(src.parent) + os.sep) and src.parent.name == "builds":
        sys.exit("refusing to write next to the release builds; use a builds-test folder")
    if not vision.exists():
        sys.exit(f"no {vision}; run the vision step first")
    out.mkdir(parents=True)
    weights = sorted(src.glob("model*.safetensors"))
    for f in weights:
        if a.copy:
            shutil.copy2(f, out / f.name)
        else:
            os.symlink(f, out / f.name)
    shutil.copy2(src / "model.safetensors.index.json", out)
    for name in SMALL:
        p = src / name if (src / name).exists() else Path(a.hf) / name
        if p.exists():
            shutil.copy2(p, out / name)
    for name in ("sensenova_loop.py", "vision.py", "processor.py"):
        shutil.copy2(HERE / "sensenova_mlx" / name, out / name)
    if a.copy:
        shutil.copy2(vision, out / "vision.safetensors")
    else:
        os.symlink(vision, out / "vision.safetensors")
    c = json.load(open(src / "config.json"))
    hf = json.load(open(Path(a.hf) / "config.json"))
    for k in VISION_KEYS:
        if k in hf:
            c[k] = hf[k]
    c["model_file"] = "sensenova_loop.py"
    meta = dict(c.get("sensenova_mlx") or {})
    meta.update(text_only=False, vision_file="vision.safetensors", vision_dtype="bfloat16",
                default_max_pixels=1048576, text_build=src.name)
    c["sensenova_mlx"] = meta
    json.dump(c, open(out / "config.json", "w"), indent=2)
    print(f"build: {out} ({len(weights)} weight files {'copied' if a.copy else 'linked'}, vision linked)")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("vision")
    p.add_argument("--hf", required=True)
    p.add_argument("--out", required=True)
    p = sp.add_parser("build")
    p.add_argument("--src", required=True)
    p.add_argument("--hf", required=True)
    p.add_argument("--vision", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--copy", action="store_true", help="copy weights instead of symlinking")
    a = ap.parse_args()
    {"vision": cmd_vision, "build": cmd_build}[a.cmd](a)


if __name__ == "__main__":
    main()
