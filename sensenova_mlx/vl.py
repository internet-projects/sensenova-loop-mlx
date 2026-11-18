# Copyright © 2026 NetProjects (Paris Michalas). Apache-2.0.
# Image + text runtime for our MLX SenseNova-Loop-9B builds: load a vision-enabled build folder (text weights +
# vision.safetensors), process images, build the prompt, prefill with 3-axis positions, greedy decode with rope_delta.

import json
import time
from pathlib import Path
from typing import List, Optional, Sequence

import mlx.core as mx
import numpy as np

from . import sensenova_loop as S
from .processor import DEFAULT_MAX_PIXELS, ImageProcessor, build_messages, build_prompt_ids
from .vision import VisionModel, load_vision_model


def encode_inputs(processor, tokenizer, prompt: str, images: Sequence, system: Optional[str] = None,
                  enable_thinking: Optional[bool] = None):
    """-> dict(input_ids np int64 [1, T], pixel_values np float32 [n_patches, 1536] or None, grids list)."""
    pix, grids = processor(list(images)) if images else (None, [])
    msgs = build_messages(prompt, len(grids), system=system)
    hf_tok = getattr(tokenizer, "_tokenizer", tokenizer)
    ids = build_prompt_ids(hf_tok, msgs, grids, processor.merge_size, enable_thinking=enable_thinking)
    return {"input_ids": np.array([ids], dtype=np.int64), "pixel_values": pix, "grids": [tuple(g) for g in grids]}


def mem(tag: str = "") -> str:
    s = (f"MEM {tag} active {mx.get_active_memory() / 2**30:.2f} GB, peak {mx.get_peak_memory() / 2**30:.2f} GB, "
         f"cache {mx.get_cache_memory() / 2**30:.2f} GB")
    print(s, flush=True)
    return s


class VLModel:
    def __init__(self, model, vision, processor, tokenizer, config, path):
        self.model, self.vision, self.processor, self.tokenizer = model, vision, processor, tokenizer
        self.config, self.path = config, path
        self.eos = set(config.get("eos_token_id") if isinstance(config.get("eos_token_id"), list)
                       else [config.get("eos_token_id")])

    @classmethod
    def load(cls, path, dtype=None, max_pixels: Optional[int] = DEFAULT_MAX_PIXELS, vision_file: Optional[str] = None,
             lazy=False):
        """dtype=mx.float32 casts text (non-quantized builds) and vision to float32 for parity runs."""
        from mlx.utils import tree_map
        from mlx_lm.utils import load_model, load_tokenizer

        path = Path(path)
        config = json.load(open(path / "config.json"))
        # with a dtype override, load lazily so the bf16 source tensors are freed as the cast is evaluated
        model, _ = load_model(path, lazy=lazy or dtype is not None, get_model_classes=lambda config: (S.Model, S.ModelArgs),
                              model_config={"model_file": None})  # always this package, not the build's copy
        if dtype is not None:
            if "quantization" in config:
                raise ValueError("dtype override is for non-quantized builds only")
            model.update(tree_map(lambda p: p.astype(dtype) if p.dtype in (mx.bfloat16, mx.float16) else p,
                                  model.parameters()))
        vf = vision_file or str(path / config.get("sensenova_mlx", {}).get("vision_file", "vision.safetensors"))
        vision = load_vision_model(config["vision_config"], vf, dtype=dtype)
        if dtype is not None:
            from mlx.utils import tree_flatten
            for _, v in tree_flatten(model.parameters()):  # one tensor at a time: peak stays ~ the fp32 size
                mx.eval(v)
        mx.eval(model.parameters(), vision.parameters())
        tok = load_tokenizer(path, eos_token_ids=config.get("eos_token_id"))
        proc = ImageProcessor.from_dir(str(path), max_pixels=max_pixels)
        return cls(model, vision, proc, tok, config, path)

    @classmethod
    def load_hf(cls, hf_dir, dtype=mx.float32, max_pixels: Optional[int] = DEFAULT_MAX_PIXELS):
        """Parity loader: build the MLX model straight from the PyTorch checkpoint, casting every tensor to `dtype`
        BEFORE sanitize. sanitize adds 1 to the RMSNorm weights (HF stores w, applies 1 + w in float32 at run time);
        our bf16 builds store (1 + w) rounded to bf16 (rel. error ~2e-3 per norm weight, 81 tensors), which is fine
        for bf16 inference but is not an fp32 reference. No tokenizer (inputs come pre-tokenized)."""
        import glob

        from mlx.utils import tree_flatten

        hf_dir = Path(hf_dir)
        config = json.load(open(hf_dir / "config.json"))
        model = S.Model(S.ModelArgs.from_dict(config))
        files = sorted(glob.glob(str(hf_dir / "model*.safetensors")))
        raw = {}
        for f in files:
            raw.update(mx.load(f))  # lazy: tensors are read when evaluated
        vis = {k: raw.pop(k) for k in list(raw) if k.startswith("model.visual.")}
        # drop the dict of source tensors so each bf16 tensor is freed once its float32 copy is evaluated
        text = {k: v.astype(dtype) if v.dtype in (mx.bfloat16, mx.float16) else v for k, v in raw.items()}
        del raw
        model.load_weights(list(model.sanitize(text).items()), strict=True)
        del text
        for _, v in tree_flatten(model.parameters()):  # one tensor at a time: peak ~ the float32 size
            mx.eval(v)
        vis = {k: v.astype(dtype) for k, v in VisionModel.sanitize(vis).items()}
        vision = VisionModel(config["vision_config"])
        vision.load_weights(list(vis.items()), strict=True)
        mx.eval(vision.parameters())
        del vis
        proc = ImageProcessor.from_dir(str(hf_dir), max_pixels=max_pixels)
        return cls(model, vision, proc, None, config, hf_dir)

    # ---- inputs ----
    def encode(self, prompt: str, images: Sequence, system: Optional[str] = None,
               enable_thinking: Optional[bool] = None):
        return encode_inputs(self.processor, self.tokenizer, prompt, images, system, enable_thinking)

    def image_features(self, pixel_values, grids):
        if pixel_values is None or not len(grids):
            return None
        f = self.vision(mx.array(pixel_values), grids)
        mx.eval(f)
        return f

    def embeddings(self, inputs, features=None):
        if features is None:
            features = self.image_features(inputs["pixel_values"], inputs["grids"])
        return self.model.prepare_image_prompt(inputs["input_ids"], features, inputs["grids"],
                                               self.vision.config.spatial_merge_size)

    # ---- forward ----
    def prefill(self, emb, cache, step: int = 2048, keep_last: int = 1):
        """Chunked prefill with cache; the LM head runs only on the last `keep_last` positions."""
        lm = self.model.language_model
        T = emb.shape[1]
        tail = []
        for s in range(0, T, step):
            h = lm.model(None, cache, input_embeddings=emb[:, s : s + step])
            if s + step >= T - keep_last + 1 or s + step >= T:
                tail.append(h[:, -min(keep_last, h.shape[1]):])
                mx.eval([c.state for c in cache] + tail)
            else:
                mx.eval([c.state for c in cache])
        h = mx.concatenate(tail, axis=1)[:, -keep_last:] if len(tail) > 1 else tail[0][:, -keep_last:]
        return lm.lm_head(h)

    def generate(self, inputs, max_tokens=256, prefill_step=2048, forced: Optional[List[int]] = None,
                 keep_logits=False, mem_every=64, verbose=True):
        """Greedy. forced: teacher-force these tokens (logits still recorded). Returns dict."""
        t0 = time.time()
        feats = self.image_features(inputs["pixel_values"], inputs["grids"])
        t_vis = time.time() - t0
        if verbose:
            mem("after vision")
        emb = self.embeddings(inputs, feats)
        cache = self.model.make_cache()
        t1 = time.time()
        y = self.prefill(emb, cache, prefill_step)[:, -1]
        mx.eval(y)
        t_pre = time.time() - t1
        if verbose:
            mem("after prefill")
        toks, logits = [], []
        n = len(forced) if forced is not None else max_tokens
        t2 = time.time()
        for i in range(n):
            if keep_logits:
                logits.append(np.array(y[0].astype(mx.float32)))
            t = int(mx.argmax(y, -1).item())
            toks.append(t)
            nxt = forced[i] if forced is not None else t
            if forced is None and t in self.eos:
                break
            y = self.model(mx.array([[nxt]]), cache=cache)[:, -1]
            if verbose and mem_every and (i + 1) % mem_every == 0:
                mem(f"decode {i + 1}")
        t_dec = time.time() - t2
        self.model.clear_mrope()
        T = inputs["input_ids"].shape[1]
        return {"tokens": toks, "logits": logits, "prompt_tokens": T,
                "image_tokens": int((inputs["input_ids"] == self.model.args.image_token_id).sum()),
                "vision_s": t_vis, "prefill_s": t_pre, "decode_s": t_dec}

    def decode_text(self, toks):
        return self.tokenizer.decode([t for t in toks if t not in self.eos])
