# Copyright © 2026 NetProjects (Paris Michalas). Apache-2.0.
# numpy (+ Pillow for decode/resize) image processor and prompt builder for SenseNova-Loop-9B image input.
# Reproduces transformers' Qwen2VLImageProcessor (PIL backend) + Qwen3VLProcessor with the settings of the release
# processor_config.json: smart_resize to multiples of patch*merge (32), min/max pixels, PIL bicubic, rescale 1/255,
# mean = std = 0.5, still image repeated to temporal_patch_size (2) frames, patches in 2x2 merge-block order with
# per-patch layout (C, T, ph, pw). Written from transformers 5.18 / 5.23 (image_processing_pil_qwen2_vl.py,
# image_transforms.py, processing_qwen3_vl.py).
#
# One deliberate difference: max_pixels defaults to DEFAULT_MAX_PIXELS (1,048,576 = 1024 x 1024 -> <= 1,024 LM tokens)
# instead of the checkpoint's 16,777,216 (16,384 tokens, 65,536 patches in one full-attention vision pass). Pass
# max_pixels=None to use the checkpoint value (needed for exact parity on images above 1 MP).

import json
import math
import os
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

DEFAULT_MAX_PIXELS = 1024 * 1024
IMAGE_PAD = "<|image_pad|>"
VISION_START = "<|vision_start|>"
VISION_END = "<|vision_end|>"


def smart_resize(height: int, width: int, factor: int = 32, min_pixels: int = 65536, max_pixels: int = 16777216):
    """transformers.models.qwen2_vl.image_processing_qwen2_vl.smart_resize (same code)."""
    if max(height, width) / min(height, width) > 200:
        raise ValueError(f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}")
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def to_rgb_array(image) -> np.ndarray:
    """PIL image / path / HWC uint8 array -> HWC uint8 RGB; transparency composited on white (HF convert_to_rgb)."""
    from PIL import Image

    if isinstance(image, (str, os.PathLike)):
        image = Image.open(image)
        image.load()
    if isinstance(image, np.ndarray):
        a = image
        if a.ndim == 2:
            a = np.stack([a] * 3, -1)
        if a.shape[-1] == 4:
            image = Image.fromarray(a.astype(np.uint8), "RGBA")
        else:
            return np.ascontiguousarray(a[..., :3].astype(np.uint8))
    has_alpha = "A" in image.getbands() or image.info.get("transparency") is not None
    if has_alpha:
        image = image.convert("RGBA")
        bg = Image.new("RGBA", image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(bg, image).convert("RGB")
    else:
        image = image.convert("RGB")
    return np.array(image)


def resize_rgb(rgb: np.ndarray, height: int, width: int, resample: int = 3) -> np.ndarray:
    """PIL resize of a uint8 HWC image (HF PIL backend: to_pil_image -> Image.resize((w, h), resample))."""
    from PIL import Image

    return np.array(Image.fromarray(rgb).resize((width, height), resample=resample))


def normalize(rgb: np.ndarray, rescale_factor: float, mean: Sequence[float], std: Sequence[float]) -> np.ndarray:
    """HWC uint8 -> CHW float32: (x * rescale in float64 -> float32 - mean) / std (HF rescale + normalize)."""
    x = (rgb.astype(np.float64) * rescale_factor).astype(np.float32)
    x = (x - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    return np.ascontiguousarray(x.transpose(2, 0, 1))


def patchify(chw: np.ndarray, patch_size: int, merge_size: int, temporal_patch_size: int):
    """CHW float32 -> [grid_h * grid_w, C * T * p * p] in 2x2 merge-block order (HF Qwen2VL patchify)."""
    C, H, W = chw.shape
    gh, gw = H // patch_size, W // patch_size
    p = chw.reshape(C, gh // merge_size, merge_size, patch_size, gw // merge_size, merge_size, patch_size)
    p = p.transpose(1, 4, 2, 5, 0, 3, 6)  # (gh/m, gw/m, m, m, C, ph, pw)
    p = np.broadcast_to(p[:, :, :, :, :, None], (*p.shape[:5], temporal_patch_size, *p.shape[5:]))
    return np.ascontiguousarray(p.reshape(gh * gw, C * temporal_patch_size * patch_size * patch_size)), gh, gw


class ImageProcessor:
    def __init__(self, patch_size=16, merge_size=2, temporal_patch_size=2, min_pixels=65536,
                 max_pixels: Optional[int] = DEFAULT_MAX_PIXELS, checkpoint_max_pixels=16777216, resample=3,
                 rescale_factor=1 / 255, image_mean=(0.5, 0.5, 0.5), image_std=(0.5, 0.5, 0.5)):
        self.patch_size, self.merge_size, self.temporal_patch_size = patch_size, merge_size, temporal_patch_size
        self.min_pixels = min_pixels
        self.max_pixels = checkpoint_max_pixels if max_pixels is None else int(max_pixels)
        self.resample, self.rescale_factor = resample, rescale_factor
        self.image_mean, self.image_std = tuple(image_mean), tuple(image_std)

    @classmethod
    def from_dir(cls, path: str, max_pixels: Optional[int] = DEFAULT_MAX_PIXELS, min_pixels: Optional[int] = None):
        d = {}
        for name in ("processor_config.json", "preprocessor_config.json"):
            f = os.path.join(path, name)
            if os.path.exists(f):
                d = json.load(open(f))
                d = d.get("image_processor", d)
                break
        size = d.get("size", {})
        ck_max = int(size.get("longest_edge", d.get("max_pixels", 16777216)))
        return cls(patch_size=d.get("patch_size", 16), merge_size=d.get("merge_size", 2),
                   temporal_patch_size=d.get("temporal_patch_size", 2),
                   min_pixels=int(min_pixels if min_pixels is not None else size.get("shortest_edge", d.get("min_pixels", 65536))),
                   max_pixels=max_pixels, checkpoint_max_pixels=ck_max, resample=d.get("resample", 3),
                   rescale_factor=d.get("rescale_factor", 1 / 255), image_mean=d.get("image_mean", (0.5,) * 3),
                   image_std=d.get("image_std", (0.5,) * 3))

    @property
    def factor(self):
        return self.patch_size * self.merge_size

    def target_size(self, height: int, width: int) -> Tuple[int, int]:
        return smart_resize(height, width, self.factor, self.min_pixels, self.max_pixels)

    def from_resized(self, rgb_resized: np.ndarray):
        """Already-resized HWC uint8 (or float in [0, 255]) -> (patches, (1, gh, gw))."""
        chw = normalize(rgb_resized, self.rescale_factor, self.image_mean, self.image_std)
        patches, gh, gw = patchify(chw, self.patch_size, self.merge_size, self.temporal_patch_size)
        return patches, (1, gh, gw)

    def __call__(self, images) -> Tuple[np.ndarray, List[Tuple[int, int, int]]]:
        """images: list of paths / PIL images / HWC arrays -> (pixel_values [n_patches, 1536] float32, grid_thw list)."""
        if not isinstance(images, (list, tuple)):
            images = [images]
        out, grids = [], []
        for im in images:
            rgb = to_rgb_array(im)
            h, w = self.target_size(rgb.shape[0], rgb.shape[1])
            p, g = self.from_resized(resize_rgb(rgb, h, w, self.resample))
            out.append(p)
            grids.append(g)
        return np.concatenate(out, 0), grids


def num_image_tokens(grid, merge_size=2) -> int:
    t, h, w = grid
    return int(t * h * w) // (merge_size * merge_size)


def build_messages(prompt: str, n_images: int, system: Optional[str] = None, images_first: bool = True):
    content = [{"type": "image"} for _ in range(n_images)]
    content = content + [{"type": "text", "text": prompt}] if images_first else [{"type": "text", "text": prompt}] + content
    msgs = [{"role": "system", "content": system}] if system else []
    return msgs + [{"role": "user", "content": content}]


def expand_image_pads(text: str, grids, merge_size=2) -> str:
    """Replace the i-th <|image_pad|> with num_image_tokens(grid_i) copies (Qwen3VLProcessor.__call__)."""
    parts = text.split(IMAGE_PAD)
    if len(parts) - 1 != len(grids):
        raise ValueError(f"prompt has {len(parts) - 1} {IMAGE_PAD} but {len(grids)} images were given")
    out = parts[0]
    for g, rest in zip(grids, parts[1:]):
        out += IMAGE_PAD * num_image_tokens(g, merge_size) + rest
    return out


def build_prompt_ids(tokenizer, messages, grids, merge_size=2, enable_thinking: Optional[bool] = None,
                     add_generation_prompt=True) -> List[int]:
    """Chat template -> expand image pads -> token ids (no BOS is added by this tokenizer)."""
    kw = {} if enable_thinking is None else {"enable_thinking": enable_thinking}
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt, **kw)
    text = expand_image_pads(text, grids, merge_size)
    ids = tokenizer.encode(text, add_special_tokens=False) if hasattr(tokenizer, "encode") else tokenizer(text)["input_ids"]
    return list(ids)
