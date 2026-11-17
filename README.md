# SenseNova-Loop on Apple Silicon (MLX)

An MLX port of [sensenova/SenseNova-Loop-9B](https://huggingface.co/sensenova/SenseNova-Loop-9B), SenseNova's looped Qwen3.5-based model, for mlx-lm. It runs the model the way SenseNova's own code does: layers 4–27 run twice with shared weights (56 layer passes per token) with two-stream mHC routing, float32 routers that are never quantized, and SenseNova's YaRN setting for 512K-token context.

Published by [NetProjects](https://netprojects.com) (Internet Projects Ltd). Apache-2.0. Not affiliated with SenseTime / SenseNova.

| Build | Size | Same top token as SenseNova's code | KL at the last token | Greedy text identical (16 prompts) |
|---|---|---|---|---|
| [6-bit](https://huggingface.co/netprojects/SenseNova-Loop-9B-MLX-6bit) (recommended) | 7.3 GB | 96.7% | 0.0056 | 2 / 16 |
| [8-bit](https://huggingface.co/netprojects/SenseNova-Loop-9B-MLX-8bit) | 9.5 GB | 97.6% | 0.0012 | 12 / 16 |
| [4-bit](https://huggingface.co/netprojects/SenseNova-Loop-9B-MLX-4bit) | 5.1 GB | 84.0% | 0.0982 | 2 / 16 |
| [bf16](https://huggingface.co/netprojects/SenseNova-Loop-9B-MLX-bf16) | 17.9 GB | 98.6% | 0.0012 | 12 / 16 |

Measured on a MacBook Pro M5 Max (128 GB), 17 November 2026, upstream revision `329a65f`: every position of 16 prompts compared with SenseNova's PyTorch code (Transformers 5.18, bf16). Text model; the vision encoder is not part of these builds.

**Checks on this port:** 6-bit, 8-bit, 4-bit, bf16 builds answered short test prompts correctly with thinking on and off, and the 6-bit build found a pass code hidden in a 64K-token prompt.

## Benchmarks (as reported by SenseNova)

The numbers below are SenseNova's own, for the original BF16 model, copied from the [original model card](https://huggingface.co/sensenova/SenseNova-Loop-9B).

SenseNova evaluates the release on agentic, coding and data-analysis tasks. From the numbers given in their card:

| Benchmark | SenseNova-Loop-9B | SenseNova-Loop-9B-Preview | Qwen3.8-27B |
|---|---|---|---|
| AutomationBench | 36.67 | 14.33 | 36.67 |
| Terminal-Bench 2.1 | 59.55 | 40.45 | |
| GAIA | 69.90 | 37.86 | |
| AIDABench | 65.31 | | 65.01 |
| NL2Repo | 41.36 | | 44.60 |
| BIRD | 72.34 | 71.38 | |

SenseNova reports that the release outperforms the Preview on all seven benchmarks in their comparison chart, which also
includes τ³-Banking and further baselines. Both use two loops by default; in SenseNova's loop-scaling study, three or four
loops scored higher still on some tasks.

The Preview builds from 7 November ([8-bit](https://huggingface.co/netprojects/SenseNova-Loop-9B-Preview-MLX-8bit), [6-bit](https://huggingface.co/netprojects/SenseNova-Loop-9B-Preview-MLX-6bit), [4-bit](https://huggingface.co/netprojects/SenseNova-Loop-9B-Preview-MLX-4bit)) stay online; the same code runs both.

## Use

Needs Python 3.11 or newer and mlx-lm 0.32 or newer (the macOS system Python 3.9 only gets an older mlx-lm that can't load this model).

```bash
pip install -U "mlx-lm>=0.32"
```

```python
from mlx_lm import load, generate

model, tokenizer = load("netprojects/SenseNova-Loop-9B-MLX-6bit", trust_remote_code=True)
messages = [{"role": "user", "content": "Explain in two sentences why the sky is blue."}]
prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False)
print(generate(model, tokenizer, prompt=prompt, max_tokens=256))
```

Server: `mlx_lm.server --model netprojects/SenseNova-Loop-9B-MLX-6bit --trust-remote-code`. Each Hugging Face repo carries `sensenova_loop.py` (from this repo) and loads it through mlx-lm's `model_file` support, so no extra package is needed.

## How it works

- **Execution plan:** layers 0–3, then 4–27 twice, then 28–31. Every visit has its own cache slot (KV cache for the 14 full-attention visits, conv + recurrent state for the Gated DeltaNet visits), as SenseNova's `_VisitCache`.
- **Two-stream mHC (layers 4–27):** the hidden state is copied into two streams at layer 4 and averaged after layer 27, on each pass. Each layer has two float32 routers (`mhc_attn`, `mhc_ffn`) giving pre / post gates and a 2×2 mixing matrix from a 20-iteration log-space Sinkhorn. The router math is compiled with `mx.compile`.
- **Long context:** YaRN (factor 2 over 262,144 positions) through mlx-lm's Qwen3.5 attention.
- **Release checkpoint:** the unused `loop_gate.logits` tensor (never read by SenseNova's forward pass) is dropped on load.
- **Precision:** routers stay float32 through conversion (kept out of the dtype cast and the quantizer); everything else as mlx-lm's Qwen3.5.

## Build and check it yourself

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
PY=.venv/bin/python bash tests/test_tiny.sh                       # tiny random models (Preview and release layout), CPU: MLX vs a PyTorch re-implementation
hf download sensenova/SenseNova-Loop-9B --revision 329a65f701ba8c48b73c30d0cbaa6f0449733673 --local-dir hf
PY=.venv/bin/python bash tests/test_tiny.sh hf                    # also vs SenseNova's own modeling_sensenova_loop.py
.venv/bin/python convert.py --hf hf --out mlx-q6 --bits 6         # or --bits 8 / 4, or no --bits for bf16
.venv/bin/python parity.py cases --hf hf --out cases.json
.venv/bin/python parity.py ref --hf hf --cases cases.json --out ref.npz --device mps
.venv/bin/python parity.py mlx --model mlx-q6 --cases cases.json --out q6.npz
.venv/bin/python parity.py compare ref.npz q6.npz
.venv/bin/python bench.py --model mlx-q6 --data <dir with the two parquet files> --out q6.json --arc 300 --gsm 50
.venv/bin/python scripts/long_ctx.py --model mlx-q6 --tokens 65536 --tokens 131072 --out longctx.json
```

The tiny test checks logits against SenseNova's code, that cached decoding equals a full forward pass, that the routers stay float32, and that switching the loop off changes the output. Run `parity.py ref` and `parity.py mlx` as separate processes.

## Files

- `sensenova_mlx/sensenova_loop.py` — the model (mlx-lm `model_type: sensenova_loop`; also works as a `model_file`)
- `convert.py` — Hugging Face folder → MLX bf16 / 8 / 6 / 5 / 4-bit (`--hi` keeps chosen modules at higher bits)
- `parity.py`, `bench.py`, `scripts/long_ctx.py` — reference comparison, the short benchmark, the long-context check
- `scripts/make_tiny.py`, `scripts/torch_loop_ref.py`, `tests/` — tiny random-weight checks

## License

Apache License 2.0 (`LICENSE`, `NOTICE`). SenseNova-Loop-9B is released by SenseNova under Apache-2.0; the MLX builds on Hugging Face are converted and quantized versions of those weights under the same license.

Credits: SenseNova / SenseTime (SenseNova-Loop), Qwen team (Qwen3.5), Apple MLX and mlx-lm. MLX port by Paris Michalas, Partner at [NetProjects](https://netprojects.com). Not affiliated with SenseTime / SenseNova.
