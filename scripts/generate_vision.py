#!/usr/bin/env python3
"""Image(s) + prompt -> text (greedy) with a vision-enabled MLX SenseNova-Loop-9B build.

  generate_vision.py --model builds-test/mlx-q6-vl --image tests/images/shapes.png \
      --prompt "Describe the shapes and their colours." --max-tokens 256 [--no-think] [--max-pixels 1048576]

MLX cache cap 2 GB (SN_CACHE_LIMIT_GB) and a memory limit (SN_MEMORY_LIMIT_GB, default 40). Prints MEM lines after
the vision tower, after prefill and every 64 decode steps.
"""
import argparse, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="vision-enabled build folder (config.json + vision.safetensors)")
    ap.add_argument("--image", action="append", default=[], help="image path (repeat for several images)")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--system", default=None)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--max-pixels", type=int, default=1048576, help="per-image cap; 0 = checkpoint value (16.7M)")
    ap.add_argument("--no-think", action="store_true", help="enable_thinking=False in the chat template")
    ap.add_argument("--prefill-step", type=int, default=2048)
    ap.add_argument("--vision-file", default=None)
    a = ap.parse_args()

    import mlx.core as mx

    mx.set_cache_limit(int(float(os.environ.get("SN_CACHE_LIMIT_GB", "2")) * 2**30))
    mx.set_memory_limit(int(float(os.environ.get("SN_MEMORY_LIMIT_GB", "40")) * 2**30))
    from sensenova_mlx.vl import VLModel, mem

    t0 = time.time()
    vl = VLModel.load(a.model, max_pixels=(a.max_pixels or None), vision_file=a.vision_file)
    print(f"loaded {a.model} in {time.time() - t0:.1f}s", flush=True)
    mem("after load")
    inputs = vl.encode(a.prompt, a.image, system=a.system, enable_thinking=(False if a.no_think else None))
    print(f"images {inputs['grids']} -> prompt {inputs['input_ids'].shape[1]} tokens", flush=True)
    r = vl.generate(inputs, max_tokens=a.max_tokens, prefill_step=a.prefill_step)
    print("=" * 40)
    print(vl.decode_text(r["tokens"]))
    print("=" * 40)
    n = len(r["tokens"])
    print(f"prompt {r['prompt_tokens']} tok ({r['image_tokens']} image) | vision {r['vision_s']:.2f}s | prefill "
          f"{r['prefill_s']:.2f}s ({r['prompt_tokens'] / max(r['prefill_s'], 1e-9):.0f} tok/s) | decode {n} tok "
          f"{n / max(r['decode_s'], 1e-9):.1f} tok/s")
    mem("end")


if __name__ == "__main__":
    main()
