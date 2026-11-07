# SenseNova-Loop on Apple Silicon (MLX)

An MLX port of [sensenova/SenseNova-Loop-9B-Preview](https://huggingface.co/sensenova/SenseNova-Loop-9B-Preview), SenseNova's looped Qwen3.5-based model, for mlx-lm. It runs the model the way SenseNova's own code does: layers 4–27 run twice with shared weights (56 layer passes per token) with two-stream mHC routing, and float32 routers that are never quantized.

Published by [NetProjects](https://netprojects.com) (Internet Projects Ltd). Apache-2.0. Not affiliated with SenseTime / SenseNova.

| Build | Size | Same top token as SenseNova's code | ARC-Challenge (300) acc / acc_norm | GSM8K (50) |
|---|---|---|---|---|
| [netprojects/SenseNova-Loop-9B-Preview-MLX-8bit](https://huggingface.co/netprojects/SenseNova-Loop-9B-Preview-MLX-8bit) (recommended) | 9.6 GB | 97.6% | 48.7 / 50.3 | 94.0 |
| [netprojects/SenseNova-Loop-9B-Preview-MLX-6bit](https://huggingface.co/netprojects/SenseNova-Loop-9B-Preview-MLX-6bit) | 7.3 GB | 97.2% | 47.7 / 49.0 | 88.0 |
| [netprojects/SenseNova-Loop-9B-Preview-MLX-4bit](https://huggingface.co/netprojects/SenseNova-Loop-9B-Preview-MLX-4bit) | 5.1 GB | 89.2% | 45.7 / 45.0 | 92.0 |
| bf16 (convert it yourself, below) | 18 GB | 98.6% | 48.7 / 51.0 | – |

Measured on a MacBook Pro M5 Max (128 GB), 7 November 2026, upstream revision `9937e91`. "Same top token" compares every position of 16 prompts with SenseNova's PyTorch code (Transformers 5.18, bf16). ARC-Challenge: first 300 test questions, log-likelihood over the four choices. GSM8K: first 50 test questions, chat template, thinking off, greedy (2 points per question). Text model; the vision encoder and the MTP head are not part of these builds.

## Use

Needs Python 3.10 or newer and mlx-lm 0.32 or newer (the macOS system Python 3.9 only gets an older mlx-lm that can't load this model).

```bash
pip install -U "mlx-lm>=0.32"
```

```python
from mlx_lm import load, generate

model, tokenizer = load("netprojects/SenseNova-Loop-9B-Preview-MLX-8bit", trust_remote_code=True)
messages = [{"role": "user", "content": "Explain in two sentences why the sky is blue."}]
prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False)
print(generate(model, tokenizer, prompt=prompt, max_tokens=256))
```

Server: `mlx_lm.server --model netprojects/SenseNova-Loop-9B-Preview-MLX-8bit --trust-remote-code`. Each Hugging Face repo carries `sensenova_loop.py` (from this repo) and loads it through mlx-lm's `model_file` support, so no extra package is needed.

## How it works

- **Execution plan:** layers 0–3, then 4–27 twice, then 28–31. Every visit has its own cache slot (KV cache for the 14 full-attention visits, conv + recurrent state for the Gated DeltaNet visits), as SenseNova's `_VisitCache`.
- **Two-stream mHC (layers 4–27):** the hidden state is copied into two streams at layer 4 and averaged after layer 27, on each pass. Each layer has two float32 routers (`mhc_attn`, `mhc_ffn`) giving pre / post gates and a 2×2 mixing matrix from a 20-iteration log-space Sinkhorn. The router math is compiled with `mx.compile`.
- **Precision:** routers stay float32 through conversion (kept out of the dtype cast and the quantizer); everything else as mlx-lm's Qwen3.5.
- **Speed:** about 18 / 20 / 22 tokens/s decode for 8 / 6 / 4-bit on an M5 Max (short runs). The second pass over layers 4–27 is most of the cost: with the loop switched off the 4-bit build runs at about 39 tokens/s, but then it is no longer the model SenseNova trained.

## Build and check it yourself

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
PY=.venv/bin/python bash tests/test_tiny.sh                       # tiny random model, CPU: MLX vs a PyTorch re-implementation
hf download sensenova/SenseNova-Loop-9B-Preview --revision 9937e91c7301b7dcbb5bf17e3e4e95c499a11e99 --local-dir hf
PY=.venv/bin/python bash tests/test_tiny.sh hf                    # also vs SenseNova's own modeling_sensenova_loop.py
.venv/bin/python convert.py --hf hf --out mlx-q8 --bits 8         # or --bits 6 / 4, or no --bits for bf16
.venv/bin/python parity.py cases --hf hf --out cases.json
.venv/bin/python parity.py ref --hf hf --cases cases.json --out ref.npz --device mps
.venv/bin/python parity.py mlx --model mlx-q8 --cases cases.json --out q8.npz
.venv/bin/python parity.py compare ref.npz q8.npz
.venv/bin/python bench.py --model mlx-q8 --data <dir with the two parquet files> --out q8.json --arc 300 --gsm 50
```

The tiny test checks logits against SenseNova's code (max difference 3e-7 on CPU), that cached decoding equals a full forward pass, that the routers stay float32, and that switching the loop off changes the output. Run `parity.py ref` and `parity.py mlx` as separate processes; the reference needs about 33 GB while loading the float32 checkpoint.

## Files

- `sensenova_mlx/sensenova_loop.py` — the model (mlx-lm `model_type: sensenova_loop`; also works as a `model_file`)
- `convert.py` — Hugging Face folder → MLX bf16 / 8 / 6 / 5 / 4-bit (`--hi` keeps chosen modules at higher bits)
- `parity.py`, `bench.py` — reference comparison and the short benchmark
- `scripts/make_tiny.py`, `scripts/torch_loop_ref.py`, `tests/` — tiny random-weight checks

## License

Apache License 2.0 (`LICENSE`, `NOTICE`). SenseNova-Loop-9B-Preview is released by SenseNova under Apache-2.0; the MLX builds on Hugging Face are converted and quantized versions of those weights under the same license.

Credits: SenseNova / SenseTime (SenseNova-Loop), Qwen team (Qwen3.5), Apple MLX and mlx-lm. MLX port by Paris Michalas, Partner at [NetProjects](https://netprojects.com). Not affiliated with SenseTime / SenseNova.
