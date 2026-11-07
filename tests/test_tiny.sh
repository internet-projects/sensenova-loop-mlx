#!/bin/bash
# Tiny random-weight checks on CPU (float32), no big download needed for part 1.
#   tests/test_tiny.sh            -> vs the PyTorch re-implementation (scripts/torch_loop_ref.py)
#   tests/test_tiny.sh <hf dir>   -> also vs SenseNova's own modeling_sensenova_loop.py from the downloaded folder
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-.venv/bin/python}"; T="$(mktemp -d)"
"$PY" scripts/make_tiny.py --out "$T/ours" 2>&1 | grep -v "falling back" || true
"$PY" scripts/make_tiny.py --out "$T/ours-noloop" --no-loop 2>&1 | grep -v "falling back" || true
echo "== MLX vs PyTorch re-implementation"
"$PY" tests/test_tiny.py "$T/ours" "$T/ours-noloop"
if [ $# -ge 1 ]; then
  echo "== MLX vs SenseNova's remote code ($1)"
  "$PY" scripts/make_tiny.py --ref remote --src "$1" --out "$T/remote" 2>&1 | grep -v "falling back" || true
  "$PY" scripts/make_tiny.py --ref remote --src "$1" --out "$T/remote-noloop" --no-loop 2>&1 | grep -v "falling back" || true
  "$PY" tests/test_tiny.py "$T/remote" "$T/remote-noloop"
fi
rm -rf "$T"
