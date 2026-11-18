#!/usr/bin/env python3
"""MTP head parity, PyTorch side (CPU, float32): SenseNova's SenseNovaLoopMTP (vendored) on the hidden states and
embeddings saved by mtp_parity_mlx.py; compares hidden outputs and top-1 through the checkpoint's lm_head.
    mtp_parity_torch.py --hf HF_DIR --mtp mtp.safetensors --mlx mlx.npz"""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ref"))
ap = argparse.ArgumentParser(); ap.add_argument("--hf", required=True); ap.add_argument("--mtp", required=True)
ap.add_argument("--mlx", required=True); a = ap.parse_args()
import numpy as np, torch
from safetensors.torch import load_file
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from sensenova_mtp_ref import SenseNovaLoopMTP
torch.set_grad_enabled(False)
cfg = json.load(open(os.path.join(a.hf, "config.json")))["text_config"]
tc = Qwen3_5TextConfig(**cfg)
m = SenseNovaLoopMTP(tc).float().eval()
# standalone use: set the attention implementation the full model would pass down (else HF picks a wrong default)
m.config._attn_implementation = "eager"; m.layers[0].self_attn.config._attn_implementation = "eager"
sd = {k[4:]: v.float() for k, v in load_file(a.mtp).items()}
missing, unexpected = m.load_state_dict(sd, strict=False)
print("missing", [k for k in missing if "rotary" not in k], "unexpected", unexpected)
idx = json.load(open(os.path.join(a.hf, "model.safetensors.index.json")))["weight_map"]
lm_head = load_file(os.path.join(a.hf, idx["lm_head.weight"]))["lm_head.weight"].float()
d = np.load(a.mlx)
n = len([k for k in d.files if k.startswith("ids")])
agree = tot = 0; cos = []; match_ref = match_mlx = 0
for i in range(n):
    h = torch.from_numpy(d[f"h{i}"])[None]; emb = torch.from_numpy(d[f"emb{i}"])[None]
    P = h.shape[1]
    pos = torch.arange(1, P).view(1, 1, -1).expand(3, 1, -1)
    out, _ = m(h[:, :-1], emb[:, 1:], pos)
    ref_top = (out[0] @ lm_head.T).argmax(-1).numpy()
    mlx_top = d[f"top{i}"]; mh = torch.from_numpy(d[f"mh{i}"])
    c = torch.nn.functional.cosine_similarity(out[0], mh, dim=-1)
    cos.append(float(c.mean())); agree += int((ref_top == mlx_top).sum()); tot += len(ref_top)
    tgt = d[f"tgt_next2_{i}"]  # MTP position j (token j+1 input) predicts token j+2
    match_ref += int((ref_top[:-1] == tgt).sum()); match_mlx += int((mlx_top[:-1] == tgt).sum())
    print(f"prompt {i}: P={P} top1 agree {int((ref_top == mlx_top).sum())}/{len(ref_top)} cos mean {float(c.mean()):.5f} min {float(c.min()):.5f}")
print(f"MTP-PARITY top1 agree {agree}/{tot} = {100*agree/tot:.1f}% | cos mean {np.mean(cos):.5f} | "
      f"teacher-forced draft accuracy ref {100*match_ref/(tot-n):.1f}% mlx {100*match_mlx/(tot-n):.1f}%")
