#!/usr/bin/env python3
"""Tiny random-weight checks for image input (CPU, float32). Needs torch + transformers + mlx (the .venv), and
SenseNova's modeling_sensenova_loop.py from the downloaded folder:

    python tests/test_vision_tiny.py <hf dir with modeling_sensenova_loop.py>

1. text-only path unchanged: new sensenova_loop.py vs the v0.2.0 file (git HEAD) -> bit-identical logits, full + cached
2. get_rope_index (numpy) == transformers Qwen3_5Model.get_rope_index on 1- and 2-image prompts (+ text after)
3. vision tower: MLX vs transformers Qwen3_5VisionModel, two images of different grids, rel-L2
4. full model with images: MLX (vision + embed scatter + mRoPE) vs SenseNova remote code: prefill logits, then
   4 cached greedy decode steps (exercises rope_delta), plus a chunked prefill split inside the image
Tiny text rope: head_dim 32, partial 0.25 -> 8 rotary dims, interleaved mRoPE section [2, 1, 1] (t, h and w all used),
YaRN factor 2.
"""
import json, os, shutil, subprocess, sys, tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mlx.core as mx

mx.set_default_device(mx.cpu)
mx.set_cache_limit(2 * 2**30)
mx.set_memory_limit(8 * 2**30)
from mlx_lm.utils import load_model

from sensenova_mlx import sensenova_loop as S
from sensenova_mlx.vision import VisionModel, load_vision_weights

TOL = float(os.environ.get("TINY_TOL", "1e-4"))
fails = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        fails.append(name)


def rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


IMG, VID, VS, VE = 500, 501, 502, 503
TEXT = dict(model_type="qwen3_5_text", hidden_size=64, intermediate_size=128, num_hidden_layers=8,
            num_attention_heads=4, num_key_value_heads=2, head_dim=32, linear_num_key_heads=2,
            linear_num_value_heads=4, linear_key_head_dim=16, linear_value_head_dim=16, linear_conv_kernel_dim=4,
            vocab_size=512, max_position_embeddings=32, rms_norm_eps=1e-6, full_attention_interval=4,
            attn_output_gate=True, attention_bias=False, tie_word_embeddings=False, hidden_act="silu",
            rope_parameters=dict(rope_type="yarn", factor=2.0, original_max_position_embeddings=16,
                                 rope_theta=10000000, partial_rotary_factor=0.25, mrope_section=[2, 1, 1],
                                 mrope_interleaved=True),
            partial_rotary_factor=0.25, pad_token_id=0, eos_token_id=1, mlp_only_layers=[], dtype="float32")
TEXT["layer_types"] = ["full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(8)]
LOOP = dict(start=2, end=6, eval_count=2, counts=[2])
MHC = dict(start=2, end=6, streams=2, iterations=20, pre_scale=1.0, fixed_identity=False)
VISION = dict(model_type="qwen3_5_vision", depth=2, hidden_size=32, intermediate_size=64, num_heads=2, in_channels=3,
              patch_size=16, spatial_merge_size=2, temporal_patch_size=2, out_hidden_size=64,
              num_position_embeddings=64, hidden_act="gelu_pytorch_tanh", deepstack_visual_indexes=[])


def make_hf(out, src):
    from transformers import AutoConfig, AutoModelForCausalLM
    sys.path.insert(0, str(ROOT / "scripts"))
    from make_tiny import mhc_bias, write_tokenizer
    cfg = dict(architectures=["SenseNovaLoopForConditionalGeneration"], model_type="sensenova_loop",
               text_config=TEXT, vision_config=VISION, loop_settings=LOOP, mhc=MHC, tie_word_embeddings=False,
               hidden_size=64, image_token_id=IMG, video_token_id=VID, vision_start_token_id=VS,
               vision_end_token_id=VE, dtype="float32")
    shutil.copy(os.path.join(src, "modeling_sensenova_loop.py"), out)
    cfg["auto_map"] = json.load(open(os.path.join(src, "config.json")))["auto_map"]
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=1)
    torch.manual_seed(0)
    c = AutoConfig.from_pretrained(out, trust_remote_code=True)
    m = AutoModelForCausalLM.from_config(c, trust_remote_code=True).float().eval()
    with torch.no_grad():
        for n_, p in m.named_parameters():
            if ".mhc_" in n_ and n_.endswith("weight"):
                p.normal_(0, 0.05)
            elif ".mhc_" in n_ and n_.endswith("bias"):
                p.copy_(mhc_bias(2) + torch.randn(8) * 0.1)
            elif "language_model" in n_ and (n_.endswith("layernorm.weight") or n_.endswith(".norm.weight")
                                             or "q_norm" in n_ or "k_norm" in n_):
                p.normal_(0, 0.1)
            elif "visual" in n_:  # default init leaves biases / norms trivial; make every term count
                if n_.endswith("bias"):
                    p.normal_(0, 0.05)
                elif "norm" in n_:
                    p.copy_(1 + 0.1 * torch.randn_like(p))
                else:
                    p.normal_(0, 0.1)
    m.save_pretrained(out, safe_serialization=True)
    write_tokenizer(out, TEXT["vocab_size"])
    return m


def load_mlx(d, module=S):
    m, _ = load_model(Path(d), get_model_classes=lambda config: (module.Model, module.ModelArgs))
    return m


def main():
    src = sys.argv[1]
    tmp = tempfile.mkdtemp(prefix="sn-vis-tiny-")
    try:
        run(src, tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("ALL PASS" if not fails else f"FAILED: {fails}")
    sys.exit(1 if fails else 0)


def run(src, tmp):
    hf = make_hf(tmp, src)
    files = sorted(str(p) for p in Path(tmp).glob("*.safetensors"))

    # ---- 1. text-only path unchanged vs v0.2.0 ----
    old_src = subprocess.run(["git", "-C", str(ROOT), "show", "HEAD:sensenova_mlx/sensenova_loop.py"],
                             capture_output=True, text=True, check=True).stdout
    oldp = os.path.join(tmp, "sensenova_loop_v020.py")
    open(oldp, "w").write(old_src)
    import importlib.util
    spec = importlib.util.spec_from_file_location("sensenova_loop_v020", oldp)
    O = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(O)
    m_old, m_new = load_mlx(tmp, O), load_mlx(tmp, S)
    ids = mx.array(np.random.default_rng(1).integers(2, 400, (2, 21)))
    a, b = m_old(ids), m_new(ids)
    same_full = bool(mx.array_equal(a, b).item())
    co, cn = m_old.make_cache(), m_new.make_cache()
    steps_same = bool(mx.array_equal(m_old(ids[:, :9], cache=co), m_new(ids[:, :9], cache=cn)).item())
    for t in range(9, 14):
        steps_same &= bool(mx.array_equal(m_old(ids[:, t:t + 1], cache=co), m_new(ids[:, t:t + 1], cache=cn)).item())
    check("text-only logits bit-identical to v0.2.0 (full + cached)", same_full and steps_same)

    # ---- 2. get_rope_index ----
    grids = [(1, 4, 6), (1, 6, 4)]
    n1, n2 = 6, 6

    def prompt(gs):
        ids = [7, 8, 9]
        for g in gs:
            ids += [VS] + [IMG] * (g[0] * g[1] * g[2] // 4) + [VE, 10, 11]
        return ids + [12, 13, 14, 15]

    ok = True
    for gs in ([grids[0]], grids, [(1, 2, 8)]):
        p = np.array([prompt(gs)])
        ours, od = S.get_rope_index(p, gs, IMG, 2)
        tt = torch.tensor(p)
        ref, rd = hf.model.get_rope_index(tt, (tt == IMG).int(), image_grid_thw=torch.tensor(gs))
        ok &= np.array_equal(ours, ref.numpy()) and np.array_equal(od, rd.numpy().reshape(-1))
    check("get_rope_index == transformers (positions + deltas)", ok)

    # ---- 3. vision tower ----
    vm = VisionModel(VISION)
    vw = VisionModel.sanitize(load_vision_weights(files))
    vm.load_weights(list(vw.items()), strict=True)
    rng = np.random.default_rng(2)
    npatch = sum(g[0] * g[1] * g[2] for g in grids)
    pix = rng.standard_normal((npatch, 3 * 2 * 16 * 16)).astype(np.float32)
    ours = np.array(vm(mx.array(pix), grids))
    with torch.no_grad():
        ref = hf.model.visual(torch.tensor(pix), grid_thw=torch.tensor(grids)).pooler_output.numpy()
    r = rel(ours, ref)
    check("vision tower vs transformers (2 images)", r < TOL and ours.shape == ref.shape, f"rel-L2 {r:.2e} {ours.shape}")

    # ---- 4. full model with images ----
    m = m_new
    p = prompt(grids)
    ids_np = np.array([p])
    feats = vm(mx.array(pix), grids)
    emb = m.prepare_image_prompt(ids_np, feats, grids)
    ours = np.array(m(None, input_embeddings=emb))
    tt = torch.tensor(ids_np)
    kw = dict(pixel_values=torch.tensor(pix), image_grid_thw=torch.tensor(grids), mm_token_type_ids=(tt == IMG).int())
    with torch.no_grad():
        out = hf(input_ids=tt, use_cache=True, **kw)
    ref = out.logits.numpy()
    r = rel(ours, ref)
    check("prefill logits with 2 images vs remote code", r < TOL, f"rel-L2 {r:.2e} delta {int(np.array(m.rope_deltas)[0])}")
    # cached greedy decode, 4 steps (positions = offset + rope_delta)
    cache = m.make_cache()
    emb = m.prepare_image_prompt(ids_np, feats, grids)
    y = m(None, cache=cache, input_embeddings=emb)[:, -1]
    past = out.past_key_values
    nxt = int(np.argmax(ref[0, -1]))
    dec_ok, worst, seq = True, 0.0, []
    for _ in range(4):
        ot = int(mx.argmax(y, -1).item())
        dec_ok &= ot == nxt
        seq.append(nxt)
        y = m(mx.array([[nxt]]), cache=cache)[:, -1]
        with torch.no_grad():
            o = hf(input_ids=torch.tensor([[nxt]]), past_key_values=past, use_cache=True)
        past = o.past_key_values
        worst = max(worst, rel(np.array(y), o.logits[:, -1].numpy()))
        nxt = int(np.argmax(o.logits[0, -1].numpy()))
    check("4 cached decode steps vs remote code", dec_ok and worst < TOL, f"max rel-L2 {worst:.2e} tokens {seq}")
    # full forward of prompt + decoded tokens (no cache) must equal the cached path (text after image)
    full_ids = np.array([p + seq])
    emb = m.prepare_image_prompt(full_ids, feats, grids)
    fo = np.array(m(None, input_embeddings=emb))
    tt2 = torch.tensor(full_ids)
    with torch.no_grad():
        fr = hf(input_ids=tt2, use_cache=False, pixel_values=torch.tensor(pix), image_grid_thw=torch.tensor(grids),
                mm_token_type_ids=(tt2 == IMG).int()).logits.numpy()
    r = rel(fo, fr)
    check("full forward prompt+generated vs remote code", r < TOL, f"rel-L2 {r:.2e}")
    # chunked prefill, split inside the first image
    cache = m.make_cache()
    emb = m.prepare_image_prompt(ids_np, feats, grids)
    k = 6
    l1 = m(None, cache=cache, input_embeddings=emb[:, :k])
    l2 = m(None, cache=cache, input_embeddings=emb[:, k:])
    r = rel(np.concatenate([np.array(l1), np.array(l2)], 1), ref)
    check("chunked prefill (split inside image) vs remote code", r < TOL, f"rel-L2 {r:.2e}")
    # sensitivity: the test must fail if positions are wrong (1-D positions; decode without rope_delta)
    emb = m.prepare_image_prompt(full_ids, feats, grids)
    m.clear_mrope()
    r1 = rel(np.array(m(None, input_embeddings=emb)), fr)
    cache = m.make_cache()
    emb = m.prepare_image_prompt(ids_np, feats, grids)
    m(None, cache=cache, input_embeddings=emb)
    m.language_model.model.mrope.delta = mx.zeros_like(m.rope_deltas)
    y0 = np.array(m(mx.array([[seq[0]]]), cache=cache)[:, -1])
    with torch.no_grad():
        o = hf(input_ids=tt2[:, : len(p) + 1], use_cache=False, pixel_values=torch.tensor(pix),
               image_grid_thw=torch.tensor(grids), mm_token_type_ids=(tt2[:, : len(p) + 1] == IMG).int())
    r2 = rel(y0, o.logits[:, -1].numpy())
    check("sensitivity: 1-D positions / zero delta are detected", r1 > 100 * TOL and r2 > 100 * TOL,
          f"rel-L2 1-D {r1:.2e}, no-delta decode {r2:.2e}")
    # runtime (sensenova_mlx/vl.py): chunked prefill (LM head on the tail only) + greedy decode, forced tokens
    from sensenova_mlx.vl import VLModel
    vl = VLModel(m, vm, None, None, {"eos_token_id": [1]}, None)
    inp = {"input_ids": ids_np, "pixel_values": pix, "grids": grids}
    g = vl.generate(inp, max_tokens=4, prefill_step=7, keep_logits=True, verbose=False)
    r = rel(g["logits"][0], ref[0, -1])
    check("VLModel.generate (prefill step 7) vs remote code", g["tokens"] == seq and r < TOL,
          f"tokens {g['tokens']} first-step rel-L2 {r:.2e}; mRoPE cleared after: {not m.language_model.model.mrope.active}")
    # VLModel.load from a folder (lazy load + float32 cast + vision weights filtered out of a mixed file)
    vl2 = VLModel.load(tmp, dtype=mx.float32, vision_file=files[0])
    g2 = vl2.generate(inp, max_tokens=4, prefill_step=64, keep_logits=True, verbose=False)
    r = rel(g2["logits"][0], ref[0, -1])
    check("VLModel.load(folder, float32) generate vs remote code", g2["tokens"] == seq and r < TOL, f"rel-L2 {r:.2e}")
    # parity loader: VLModel.load_hf casts to float32 BEFORE sanitize adds 1 to the RMSNorm weights; loading bf16
    # weights and casting afterwards (what a bf16 build gives) rounds 1 + w to bf16 (~2e-3 rel.), which made the
    # full-size fp32 P3 look like a 1e-2 mismatch
    bdir = os.path.join(tmp, "bf16")
    os.makedirs(bdir)
    for f in os.listdir(tmp):
        if f.endswith((".json", ".py")):
            shutil.copy(os.path.join(tmp, f), bdir)
    raw = {}
    for f in files:
        raw.update({k: v.astype(mx.bfloat16) if v.dtype == mx.float32 and ".mhc_" not in k else v
                    for k, v in mx.load(f).items()})
    mx.save_safetensors(os.path.join(bdir, "model.safetensors"), raw)
    k = "model.language_model.layers.3.input_layernorm.weight"
    exact = np.array(raw[k].astype(mx.float32)) + 1.0
    vl3 = VLModel.load_hf(bdir, dtype=mx.float32)
    got = np.array(vl3.model.language_model.model.layers[3].input_layernorm.weight)
    late = np.array((raw[k] + 1.0).astype(mx.float32))  # bf16 build path
    check("load_hf: norm 1+w computed in float32", np.array_equal(got, exact) and not np.array_equal(late, exact),
          f"bf16-path max abs err {np.abs(late - exact).max():.1e}")
    # text-only prompt after clear_mrope equals the plain path again
    m.clear_mrope()
    check("clear_mrope -> text path", bool(mx.array_equal(m(ids), m_old(ids)).item()))
    # prepare_image_prompt with no images -> plain embeddings and mRoPE off
    e0 = m.prepare_image_prompt(np.array(ids), None, [])
    check("no-image prompt keeps mRoPE off", not m.language_model.model.mrope.active
          and bool(mx.array_equal(m(None, input_embeddings=e0), m_old(ids)).item()))


if __name__ == "__main__":
    main()
