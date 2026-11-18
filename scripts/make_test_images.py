#!/usr/bin/env python3
"""Synthetic, text-free test images for the vision parity / smoke runs (numpy + Pillow; deterministic).

  make_test_images.py --out tests/images
"""
import argparse, os

import numpy as np
from PIL import Image


def disk(h, w, cy, cx, r):
    y, x = np.ogrid[:h, :w]
    return (y - cy) ** 2 + (x - cx) ** 2 <= r * r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(0)

    # 1. shapes 640x480: red disk left, blue square right, green bar at the bottom, light grey background
    h, w = 480, 640
    im = np.full((h, w, 3), 230, np.uint8)
    im[disk(h, w, 200, 170, 110)] = (220, 30, 30)
    im[120:300, 380:560] = (30, 60, 220)
    im[400:440, 60:580] = (30, 170, 60)
    Image.fromarray(im).save(os.path.join(a.out, "shapes.png"))

    # 2. small RGBA 200x150 (below min_pixels -> upscaled): yellow disk on a half-transparent gradient
    h, w = 150, 200
    rgba = np.zeros((h, w, 4), np.uint8)
    rgba[..., 0] = np.linspace(0, 255, w, dtype=np.uint8)[None]
    rgba[..., 1] = np.linspace(255, 0, h, dtype=np.uint8)[:, None]
    rgba[..., 2] = 90
    rgba[..., 3] = 128
    m = disk(h, w, 75, 100, 50)
    rgba[m] = (250, 220, 20, 255)
    Image.fromarray(rgba, "RGBA").save(os.path.join(a.out, "small_rgba.png"))

    # 3. grey 333x517 (odd aspect, mode L): concentric rings
    h, w = 517, 333
    y, x = np.mgrid[:h, :w]
    r = np.sqrt((y - h / 2) ** 2 + (x - w / 2) ** 2)
    Image.fromarray(((np.sin(r / 9.0) * 0.5 + 0.5) * 255).astype(np.uint8), "L").save(os.path.join(a.out, "rings_grey.png"))

    # 4. wide 1600x900 (above the 1 MP default cap): vertical colour stripes + noise + one large orange disk
    h, w = 900, 1600
    cols = np.array([(200, 40, 40), (40, 160, 40), (40, 40, 200), (230, 230, 230)], np.uint8)
    im = cols[(np.arange(w) // 100) % 4][None].repeat(h, 0).astype(np.int16)
    im += rng.integers(-12, 13, im.shape, dtype=np.int16)
    im = np.clip(im, 0, 255).astype(np.uint8)
    im[disk(h, w, 450, 800, 260)] = (250, 140, 20)
    Image.fromarray(im).save(os.path.join(a.out, "stripes_wide.png"))

    # 5. three disks 512x512 (counting smoke): three black disks on white
    h, w = 512, 512
    im = np.full((h, w, 3), 255, np.uint8)
    for cy, cx in ((130, 130), (130, 380), (380, 256)):
        im[disk(h, w, cy, cx, 70)] = 0
    Image.fromarray(im).save(os.path.join(a.out, "three_disks.png"))
    print("images:", sorted(os.listdir(a.out)))


if __name__ == "__main__":
    main()
