#!/usr/bin/env python3
"""Text-only regression on a real build: the image-capable sensenova_loop.py must give bit-identical logits to the
v0.2.0 file (git HEAD) on text prompts, full forward and cached decode.

  text_regress.py --model mlx-q4 [--old OLD_sensenova_loop.py]
"""
import argparse, importlib.util, os, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PROMPTS = ["The capital of Greece is", "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
           "Q: A train leaves at 9:40 and arrives at 13:05. How long is the trip?\nA:", "Step-by-step: 17 * 24 ="]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--old", default=None)
    a = ap.parse_args()
    import mlx.core as mx
    mx.set_cache_limit(int(float(os.environ.get("SN_CACHE_LIMIT_GB", "2")) * 2**30))
    mx.set_memory_limit(int(float(os.environ.get("SN_MEMORY_LIMIT_GB", "30")) * 2**30))
    from mlx_lm.utils import load_model, load_tokenizer
    from sensenova_mlx import sensenova_loop as NEW

    old = a.old
    if old is None:
        src = subprocess.run(["git", "-C", str(ROOT), "show", "v0.2.0:sensenova_mlx/sensenova_loop.py"],
                             capture_output=True, text=True)
        if src.returncode:
            src = subprocess.run(["git", "-C", str(ROOT), "show", "HEAD:sensenova_mlx/sensenova_loop.py"],
                                 capture_output=True, text=True, check=True)
        old = os.path.join(tempfile.mkdtemp(), "sensenova_loop_old.py")
        open(old, "w").write(src.stdout)
    spec = importlib.util.spec_from_file_location("sensenova_loop_old", old)
    OLD = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(OLD)
    tok = load_tokenizer(Path(a.model))
    res = {}
    for name, mod in (("old", OLD), ("new", NEW)):
        m, _ = load_model(Path(a.model), get_model_classes=lambda config, mod=mod: (mod.Model, mod.ModelArgs),
                              model_config={"model_file": None})  # use OLD/NEW explicitly, not the build's copy
        outs = []
        for p in PROMPTS:
            x = mx.array([tok.encode(p)])
            full = m(x)
            cache = m.make_cache()
            y = m(x, cache=cache)[:, -1:]
            steps = [y]
            for _ in range(8):
                t = mx.argmax(y, -1)
                y = m(t, cache=cache)[:, -1:]
                steps.append(y)
            outs.append((full, mx.concatenate(steps, 1)))
            mx.eval(outs[-1])
        res[name] = outs
        print(f"MEM {name}: active {mx.get_active_memory() / 2**30:.1f} GB peak {mx.get_peak_memory() / 2**30:.1f} GB",
              flush=True)
        del m
        mx.clear_cache()
    ok = all(bool(mx.array_equal(a_[0], b_[0]).item()) and bool(mx.array_equal(a_[1], b_[1]).item())
             for a_, b_ in zip(res["old"], res["new"]))
    print(f"{'PASS' if ok else 'FAIL'} text-only regression {Path(a.model).name}: {len(PROMPTS)} prompts, full forward + "
          f"9 cached steps, bit-identical = {ok}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
