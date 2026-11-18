#!/usr/bin/env python3
"""Image-input parity for the MLX SenseNova-Loop-9B port (PyTorch on CPU only; never MPS).

Which venv: [V] = .venv-vision (MLX + Pillow, no torch), [T] = .venv (torch + transformers 5.18 + MLX, no Pillow).
Run each step as its own process; PyTorch and MLX GPU work never run at the same time.

  P0 [V]  p0        --hf HF --build B --images DIR --out D     HF Qwen3VLProcessor (PIL backend) vs ours; writes
                                                              D/inputs.npz (all prompts) + D/p0-resized.npz
  P0 [T]  p0-torch  --hf HF --dir D                           transformers 5.18 smart_resize + torch patchify vs ours
  P1 [T]  p1-ref    --hf HF --dir D                           reference vision tower fp32 CPU -> D/p1-ref.npz
  P1 [T]  p1-mlx    --vision VISION.safetensors --hf HF --dir D   MLX vision fp32 (CPU) vs ref: rel-L2 < 1e-4
  P2 [T]  p2        --hf HF --dir D                           get_rope_index: ours == transformers, exact
  P3 [T]  p3-ref    --hf HF --dir D                           full model fp32 CPU: last 8 prefill logits + 4 greedy
                                                              decode steps for the P3 prompts -> D/p3-ref.npz  (JOB)
  P3 [V]  p3-mlx    --model BF16_VL_BUILD --dir D             MLX fp32 (teacher-forced on the ref tokens) vs ref (JOB)
  P4 [V]  p4        --model VL_BUILD --name NAME --dir D      bf16 / q8 / q6 / q4: top-1 vs the bf16 run (JOB each)
"""
import argparse, ast, json, math, os, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# prompts: (id, images, question). P3 uses the first two; P4 uses all.
PROMPTS = [
    ("shapes", ["shapes.png"], "Describe the shapes in this image and their colours."),
    ("two", ["small_rgba.png", "rings_grey.png"], "Compare the two pictures. What does each one show?"),
    ("disks", ["three_disks.png"], "How many black circles are in this image? Answer with a number."),
    ("square", ["shapes.png"], "What colour is the square? Answer in one word."),
    ("wide", ["stripes_wide.png"], "What is in the centre of this image?"),
    ("rings", ["rings_grey.png"], "Describe this pattern in one sentence."),
]
N_PREFILL, N_DECODE, N_GREEDY = 8, 4, 32


def rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def load_inputs(d, text=False):
    z = np.load(os.path.join(d, "inputs.npz"))
    meta = json.load(open(os.path.join(d, "inputs.json")))
    out = []
    for i, m in enumerate(meta):
        out.append(dict(id=m["id"], grids=[tuple(g) for g in m["grids"]], input_ids=z[f"{i}_ids"],
                        pixel_values=z[f"{i}_pix"]))
    tf = os.path.join(d, "inputs-text.npz")
    if text and os.path.exists(tf):
        zt = np.load(tf)
        out.append(dict(id="text", grids=[], input_ids=zt["ids"], pixel_values=None))
    return out


TEXT_CONTROL = (
    "Read the following notes and then answer the question at the end.\n\n"
    "The harbour town of Kelmora sits at the mouth of a slow river. Its fishing fleet leaves before dawn and returns "
    "in the early afternoon, when the market opens on the stone quay. In spring the town hosts a boat race between "
    "the three districts: North Bank, Old Quay and the Terraces. Last year North Bank won by two lengths, Old Quay "
    "came second and the Terraces, whose crew had lost an oar at the first turn, finished last. The race committee "
    "has since changed the rules: every boat must now carry a spare oar, and the course was shortened from four "
    "kilometres to three so that it ends in front of the market rather than beyond the lighthouse. Fishermen say the "
    "new finish is better for spectators but worse for the crews, because the last stretch now runs against the "
    "tide. The committee also moved the race from the first to the second Sunday of May, after complaints that it "
    "clashed with the flower fair in the hills.\n\n"
    "Question: which district finished last in the previous race, why, and what rule changed because of it?"
)


def cmd_text_inputs(a):
    """[V] tokenize the text-only control prompt with the chat template -> D/inputs-text.npz"""
    from transformers import AutoTokenizer
    from sensenova_mlx.processor import build_messages, build_prompt_ids
    tok = AutoTokenizer.from_pretrained(a.build)
    ids = build_prompt_ids(tok, build_messages(TEXT_CONTROL, 0), [], enable_thinking=False)
    np.savez(os.path.join(a.dir, "inputs-text.npz"), ids=np.array([ids], np.int64))
    print(f"text control: {len(ids)} tokens -> {a.dir}/inputs-text.npz")


def regions(ids, image_token_id=248056):
    """index arrays: text before the first image, image tokens, text after the last image (or all text)."""
    ids = np.asarray(ids).reshape(-1)
    img = np.nonzero(ids == image_token_id)[0]
    T = len(ids)
    if len(img) == 0:
        return {"text": np.arange(T)}
    return {"before": np.arange(0, img[0]), "image": img,
            "between+after": np.array([i for i in range(img[0], T) if ids[i] != image_token_id])}


def diag_positions(ids, image_token_id=248056, k=6):
    """a few positions per region for the per-visit comparison."""
    out = []
    for r in regions(ids, image_token_id).values():
        if len(r):
            out += list(r[np.unique(np.linspace(0, len(r) - 1, min(k, len(r))).round().astype(int))])
    return np.array(sorted(set(int(x) for x in out)), np.int64)


# ---------------- P0 [V] ----------------
def cmd_p0(a):
    from PIL import Image
    from transformers import AutoTokenizer, ProcessorMixin
    # Qwen3VLProcessor insists on a torchvision video processor; images only here, so skip that class check.
    ProcessorMixin.check_argument_for_proper_class = lambda self, *x, **k: None
    from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import Qwen2VLImageProcessorPil
    from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor

    from sensenova_mlx.processor import ImageProcessor, build_messages, build_prompt_ids, to_rgb_array, resize_rgb

    os.makedirs(a.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.build)
    hfp = Qwen3VLProcessor(image_processor=Qwen2VLImageProcessorPil.from_pretrained(a.hf), tokenizer=tok,
                           video_processor=None, chat_template=open(os.path.join(a.hf, "chat_template.jinja")).read())
    fails = []
    imgs = sorted(f for f in os.listdir(a.images) if f.endswith(".png"))
    resized = {}
    for cap in (None, 1048576):
        ours = ImageProcessor.from_dir(a.hf, max_pixels=cap)
        kw = {} if cap is None else {"size": {"shortest_edge": 65536, "longest_edge": cap}}
        for f in imgs:
            im = Image.open(os.path.join(a.images, f))
            ref = hfp.image_processor(images=[im], return_tensors="np", **kw)
            pv, grids = ours([os.path.join(a.images, f)])
            same_grid = [list(map(int, g)) for g in ref["image_grid_thw"]] == [list(g) for g in grids]
            d = float(np.abs(pv - ref["pixel_values"]).max()) if same_grid else float("inf")
            ok = same_grid and d == 0.0
            print(f"{'PASS' if ok else 'FAIL'} P0 pixels {f} ({im.mode} {im.size[0]}x{im.size[1]}) cap {cap}: "
                  f"grid {grids[0]} max|d| {d:.1e}")
            if not ok:
                fails.append(f"pixels {f} {cap}")
            if cap is None:
                rgb = to_rgb_array(os.path.join(a.images, f))
                h, w = ours.target_size(*rgb.shape[:2])
                resized[f] = resize_rgb(rgb, h, w)
    np.savez(os.path.join(a.out, "p0-resized.npz"), **{k: v for k, v in resized.items()})

    # prompts: HF processor (chat template + expansion + tokenize) vs ours, default cap
    ours = ImageProcessor.from_dir(a.hf)
    meta, arrays = [], {}
    kw = {"size": {"shortest_edge": 65536, "longest_edge": 1048576}}
    for i, (pid, files, q) in enumerate(PROMPTS):
        paths = [os.path.join(a.images, f) for f in files]
        msgs = build_messages(q, len(files))
        pv, grids = ours(paths)
        ids = build_prompt_ids(tok, msgs, grids, enable_thinking=False)
        text = hfp.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        ref = hfp(text=[text], images=[Image.open(p) for p in paths], return_tensors="np", **kw)
        same = list(ref["input_ids"][0]) == ids and float(np.abs(ref["pixel_values"] - pv).max()) == 0.0 \
            and int(ref["mm_token_type_ids"].sum()) == ids.count(248056)
        print(f"{'PASS' if same else 'FAIL'} P0 prompt {pid}: {len(ids)} tokens, {ids.count(248056)} image tokens, "
              f"grids {grids}")
        if not same:
            fails.append(f"prompt {pid}")
        meta.append({"id": pid, "images": files, "question": q, "grids": [list(g) for g in grids]})
        arrays[f"{i}_ids"] = np.array([ids], np.int64)
        arrays[f"{i}_pix"] = pv
    np.savez(os.path.join(a.out, "inputs.npz"), **arrays)
    json.dump(meta, open(os.path.join(a.out, "inputs.json"), "w"), indent=1)
    print("P0 ALL PASS" if not fails else f"P0 FAILED: {fails}")
    sys.exit(1 if fails else 0)


# ---------------- P0 [T] ----------------
def _hf_funcs(path, names, extra_globals):
    """Pull top-level functions / class methods out of a transformers source file without importing it."""
    src = open(path).read()
    tree = ast.parse(src)
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in names:
            mod = ast.Module(body=[node], type_ignores=[])
            ns = dict(extra_globals)
            exec(compile(mod, path, "exec"), ns)
            out[node.name] = ns[node.name]
    return out


def cmd_p0_torch(a):
    import torch, transformers
    from sensenova_mlx.processor import ImageProcessor, smart_resize as our_resize

    base = Path(transformers.__file__).parent / "models" / "qwen2_vl" / "image_processing_qwen2_vl.py"
    f = _hf_funcs(str(base), {"smart_resize", "patchify"}, {"math": math, "torch": torch})
    fails = []
    rng = np.random.default_rng(0)
    bad = 0
    sizes = [(int(h), int(w)) for h, w in rng.integers(16, 5000, (2000, 2)) if max(h, w) / min(h, w) <= 200]
    sizes += [(150, 200), (517, 333), (900, 1600), (4096, 4096), (8000, 6000), (31, 6000)]
    for h, w in sizes:
        for mx_ in (16777216, 1048576):
            if f["smart_resize"](h, w, 32, 65536, mx_) != our_resize(h, w, 32, 65536, mx_):
                bad += 1
    print(f"{'PASS' if bad == 0 else 'FAIL'} P0 smart_resize vs transformers {transformers.__version__}: "
          f"{bad} mismatches over {2 * len(sizes)} sizes")
    if bad:
        fails.append("smart_resize")
    ours = ImageProcessor.from_dir(a.hf, max_pixels=None)
    z = np.load(os.path.join(a.dir, "p0-resized.npz"))
    for k in z.files:
        rgb = z[k]
        pv, g = ours.from_resized(rgb)
        x = torch.from_numpy(rgb.astype(np.float32)).permute(2, 0, 1)[None]
        # transformers torch path: rescale (x * 1/255) and normalize ((x - 0.5) / 0.5) in float32, then patchify
        x = (x * (1 / 255) - 0.5) / 0.5
        ref, gh, gw = f["patchify"](None, x, 16, 2, 2)
        d = float(np.abs(ref[0].numpy() - pv).max())
        ok = (gh, gw) == g[1:] and d < 1e-6
        print(f"{'PASS' if ok else 'FAIL'} P0 torch patchify {k}: grid {(gh, gw)} max|d| {d:.1e}")
        if not ok:
            fails.append(k)
    print("P0-torch ALL PASS" if not fails else f"P0-torch FAILED: {fails}")
    sys.exit(1 if fails else 0)


# ---------------- P1 ----------------
def _torch_vision(hf):
    import torch
    from safetensors.torch import load_file
    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

    cfg = AutoConfig.from_pretrained(hf, trust_remote_code=True)
    vm = Qwen3_5VisionModel._from_config(cfg.vision_config, torch_dtype=torch.float32).eval()
    idx = json.load(open(os.path.join(hf, "model.safetensors.index.json")))["weight_map"]
    sd = {}
    for s in sorted({v for k, v in idx.items() if k.startswith("model.visual.")}):
        for k, v in load_file(os.path.join(hf, s)).items():
            if k.startswith("model.visual."):
                sd[k[len("model.visual."):]] = v.float()
    vm.load_state_dict(sd, strict=True)
    return vm


def cmd_p1_ref(a):
    import torch
    torch.set_grad_enabled(False)
    vm = _torch_vision(a.hf)
    out = {}
    done = {}
    for i, p in enumerate(load_inputs(a.dir)):
        key = json.dumps(p["grids"]) + str(p["pixel_values"].shape) + str(float(p["pixel_values"].sum()))
        if key in done:
            continue
        t0 = time.time()
        r = vm(torch.from_numpy(p["pixel_values"]).float(), grid_thw=torch.tensor(p["grids"])).pooler_output
        out[p["id"]] = r.numpy().astype(np.float32)
        done[key] = 1
        print(f"P1 ref {p['id']}: {tuple(r.shape)} in {time.time() - t0:.1f}s", flush=True)
    np.savez(os.path.join(a.dir, "p1-ref.npz"), **out)


def cmd_p1_mlx(a):
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    mx.set_cache_limit(2 * 2**30)
    mx.set_memory_limit(16 * 2**30)
    from sensenova_mlx.vision import load_vision_model
    cfg = json.load(open(os.path.join(a.hf, "config.json")))["vision_config"]
    ref = np.load(os.path.join(a.dir, "p1-ref.npz"))
    v32 = load_vision_model(cfg, a.vision, dtype=mx.float32)
    # bf16 is not checked here: MLX's CPU bf16 matmul (BNNS) segfaults on these shapes (mlx 0.32.4); the bf16 tower
    # runs on the GPU in the P4 jobs.
    fails = []
    for p in load_inputs(a.dir):
        if p["id"] not in ref.files:
            continue
        r = ref[p["id"]]
        o32 = np.array(v32(mx.array(p["pixel_values"]), p["grids"]))
        e32 = rel(o32, r)
        ok = o32.shape == r.shape and e32 < 1e-4
        print(f"{'PASS' if ok else 'FAIL'} P1 vision {p['id']} {r.shape}: fp32 rel-L2 {e32:.2e} "
              f"max|d| {float(np.abs(o32 - r).max()):.2e}")
        if not ok:
            fails.append(p["id"])
    print("P1 ALL PASS" if not fails else f"P1 FAILED: {fails}")
    sys.exit(1 if fails else 0)


# ---------------- P2 [T] ----------------
def cmd_p2(a):
    import types, torch
    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model
    from sensenova_mlx.sensenova_loop import get_rope_index

    cfg = AutoConfig.from_pretrained(a.hf, trust_remote_code=True)
    fake = types.SimpleNamespace(config=cfg)
    fake.get_vision_position_ids = types.MethodType(Qwen3_5Model.get_vision_position_ids, fake)
    fails = []
    for p in load_inputs(a.dir):
        ids = torch.from_numpy(p["input_ids"])
        rp, rd = Qwen3_5Model.get_rope_index(fake, ids, (ids == cfg.image_token_id).int(),
                                             image_grid_thw=torch.tensor(p["grids"]))
        op, od = get_rope_index(p["input_ids"], p["grids"], cfg.image_token_id, cfg.vision_config.spatial_merge_size)
        ok = np.array_equal(op, rp.numpy()) and np.array_equal(od, rd.numpy().reshape(-1))
        print(f"{'PASS' if ok else 'FAIL'} P2 positions {p['id']}: T={ids.shape[1]} rope_delta {int(od[0])} "
              f"max pos {int(op.max())}")
        if not ok:
            fails.append(p["id"])
    print("P2 ALL PASS" if not fails else f"P2 FAILED: {fails}")
    sys.exit(1 if fails else 0)


# ---------------- P3 ----------------
def _cases(a):
    return tuple(x for x in a.cases.split(",") if x)


def cmd_p3_ref(a):
    """per case -> D/p3-ref-<case>.npz: last 8 prefill logits, 4 greedy decode logits + tokens, rope_delta,
    final (normed) hidden at every prompt position, and every visit's hidden state at a few positions per region."""
    import torch
    from transformers import AutoModelForCausalLM
    torch.set_grad_enabled(False)
    t0 = time.time()
    m = AutoModelForCausalLM.from_pretrained(a.hf, trust_remote_code=True, dtype=torch.float32).to("cpu").eval()
    print(f"P3 ref: loaded in {time.time() - t0:.0f}s, visual dtype {next(m.model.visual.parameters()).dtype}", flush=True)
    for p in load_inputs(a.dir, text=True):
        if p["id"] not in _cases(a):
            continue
        t0 = time.time()
        ids = torch.from_numpy(p["input_ids"])
        kw = {}
        if p["pixel_values"] is not None:
            kw = dict(pixel_values=torch.from_numpy(p["pixel_values"]), image_grid_thw=torch.tensor(p["grids"]),
                      mm_token_type_ids=(ids == m.config.image_token_id).int())
        m.model.rope_deltas = None
        o = m(input_ids=ids, use_cache=True, logits_to_keep=N_PREFILL, output_hidden_states=True, **kw)
        out = {}
        sel = diag_positions(p["input_ids"], m.config.image_token_id)
        hs = o.hidden_states  # embedding + one per visit (last one = after the final norm)
        out["hfinal"] = hs[-1][0].float().numpy()
        out["sel"] = sel
        for i, h in enumerate(hs):
            out[f"visit{i}"] = h[0, torch.from_numpy(sel)].float().numpy()
        pre = o.logits[0].float().numpy()
        past, toks, dec = o.past_key_values, [], []
        nxt = int(pre[-1].argmax())
        for _ in range(N_DECODE):
            toks.append(nxt)
            o = m(input_ids=torch.tensor([[nxt]]), past_key_values=past, use_cache=True)
            past = o.past_key_values
            lg = o.logits[0, -1].float().numpy()
            dec.append(lg)
            nxt = int(lg.argmax())
        rd = m.model.rope_deltas
        out.update(prefill=pre.astype(np.float32), decode=np.stack(dec).astype(np.float32),
                   tokens=np.array(toks, np.int64),
                   rope_delta=(rd.numpy().reshape(-1) if rd is not None else np.zeros(1, np.int64)))
        np.savez(os.path.join(a.dir, f"p3-ref-{p['id']}.npz"), **out)
        print(f"P3 ref {p['id']}: T={ids.shape[1]} tokens {toks} rope_delta {int(out['rope_delta'][0])} "
              f"visits {len(hs) - 1} in {time.time() - t0:.0f}s", flush=True)


def _ref_for(d, pid):
    f = os.path.join(d, f"p3-ref-{pid}.npz")
    if os.path.exists(f):
        return dict(np.load(f))
    old = np.load(os.path.join(d, "p3-ref.npz"))  # job 01 format: logits only
    return {k[len(pid) + 1:]: old[k] for k in old.files if k.startswith(pid + "_")}


def cmd_p3_mlx(a):
    import mlx.core as mx
    mx.set_cache_limit(int(float(os.environ.get("SN_CACHE_LIMIT_GB", "2")) * 2**30))
    mx.set_memory_limit(int(float(os.environ.get("SN_MEMORY_LIMIT_GB", "48")) * 2**30))
    from sensenova_mlx.vl import VLModel, mem
    if a.hf_weights:
        vl = VLModel.load_hf(a.hf_weights, dtype=mx.float32)
        src = "fp32 from the HF checkpoint"
    else:
        vl = VLModel.load(a.model, dtype=mx.float32)
        src = "bf16 build cast to fp32 (norm weights carry bf16 rounding of 1+w)"
    mem("loaded")
    print(f"P3 MLX weights: {src}; tolerance rel-L2 {a.tol:g}", flush=True)
    tm = vl.model.language_model.model
    fails = []
    for p in load_inputs(a.dir, text=True):
        if p["id"] not in _cases(a):
            continue
        ref = _ref_for(a.dir, p["id"])
        emb = vl.embeddings(p)
        d_ours = int(np.array(vl.model.rope_deltas)[0]) if vl.model.rope_deltas is not None else 0
        cache = vl.model.make_cache()
        diag = "hfinal" in ref
        if diag:
            col = []
            tm.collect_hidden = col
            h = tm(None, cache, input_embeddings=emb)
            tm.collect_hidden = None
            sel = mx.array(ref["sel"])
            visits = [np.array(x[0][sel].astype(mx.float32)) for x in col]
            hfinal = np.array(h[0].astype(mx.float32))
            visits[-1] = hfinal[ref["sel"]]  # the reference's last entry is after the final norm
            pre = np.array(vl.model.language_model.lm_head(h[:, -N_PREFILL:])[0].astype(mx.float32))
            del col
        else:
            pre = np.array(vl.prefill(emb, cache, keep_last=N_PREFILL)[0].astype(mx.float32))
        dec = []
        for t in ref["tokens"]:
            dec.append(np.array(vl.model(mx.array([[int(t)]]), cache=cache)[0, -1].astype(mx.float32)))
        vl.model.clear_mrope()
        mem(p["id"])
        rp, rdc = ref["prefill"], ref["decode"]
        e_pre = max(rel(pre[i], rp[i]) for i in range(N_PREFILL))
        e_dec = max(rel(dec[i], rdc[i]) for i in range(N_DECODE))
        top = np.mean(np.concatenate([pre.argmax(-1) == rp.argmax(-1), np.stack(dec).argmax(-1) == rdc.argmax(-1)]))
        dmax = float(max(np.abs(pre - rp).max(), np.abs(np.stack(dec) - rdc).max()))
        ok = e_pre <= a.tol and e_dec <= a.tol and top == 1.0 and d_ours == int(ref["rope_delta"][0])
        print(f"{'PASS' if ok else 'FAIL'} P3 {p['id']}: prefill max rel-L2 {e_pre:.2e} | decode max rel-L2 {e_dec:.2e} "
              f"| max|dlogit| {dmax:.2e} | top-1 {100 * top:.0f}% | rope_delta {d_ours}", flush=True)
        if diag:
            per = np.array([rel(hfinal[i], ref["hfinal"][i]) for i in range(hfinal.shape[0])])
            iid = vl.model.args.image_token_id
            for name, idx in regions(p["input_ids"], iid).items():
                if len(idx):
                    q = per[idx]
                    print(f"   final hidden rel-L2, {name:14s} ({len(idx):4d} pos): mean {q.mean():.2e} "
                          f"median {np.median(q):.2e} max {q.max():.2e} (at pos {int(idx[q.argmax()])})")
            img = np.nonzero(np.asarray(p["input_ids"]).reshape(-1) == iid)[0]
            if len(img):  # a position / rope_delta error would show as a jump right after the image
                j = int(img[-1])
                w = [i for i in range(max(0, int(img[0]) - 3), min(len(per), j + 5)) if i < img[0] or i > j]
                print("   around the image (pos: rel-L2): " + ", ".join(f"{i}: {per[i]:.1e}" for i in w))
            n = len(visits)
            pick = sorted(set([0, 1, 2, 3, 4] + list(range(8, n, 8)) + [n - 2, n - 1]))
            regs = regions(p["input_ids"], iid)
            selpos = list(ref["sel"])
            for name, idx in regs.items():
                cols = [selpos.index(i) for i in idx if i in selpos]
                if not cols:
                    continue
                vals = []
                for v in pick:
                    a_, b_ = visits[v][cols].reshape(len(cols), -1), ref[f"visit{v}"][cols].reshape(len(cols), -1)
                    vals.append(f"{v}:{max(rel(a_[c], b_[c]) for c in range(len(cols))):.1e}")
                print(f"   per-visit hidden max rel-L2, {name} ({len(cols)} pos): " + " ".join(vals))
        if not ok:
            fails.append(p["id"])
    print("P3 ALL PASS" if not fails else f"P3 FAILED: {fails}")
    sys.exit(1 if fails else 0)


# ---------------- P4 [V] ----------------
def cmd_p4(a):
    import mlx.core as mx
    mx.set_cache_limit(int(float(os.environ.get("SN_CACHE_LIMIT_GB", "2")) * 2**30))
    mx.set_memory_limit(int(float(os.environ.get("SN_MEMORY_LIMIT_GB", "40")) * 2**30))
    from sensenova_mlx.vl import VLModel, mem
    base_f = os.path.join(a.dir, "p4-bf16.npz")
    base = np.load(base_f) if a.name != "bf16" else None
    if a.name != "bf16" and base is None:
        sys.exit("run the bf16 P4 first")
    vl = VLModel.load(a.model)
    mem("loaded")
    out, agree, total, lines = {}, 0, 0, []
    for p in load_inputs(a.dir):
        t0 = time.time()
        emb = vl.embeddings(p)
        cache = vl.model.make_cache()
        lg = vl.prefill(emb, cache, keep_last=N_PREFILL)[0]
        top_pre = np.array(mx.argmax(lg, -1))
        y = lg[-1:]
        forced = base[f"{p['id']}_greedy"] if base is not None else None
        toks, top_dec = [], []
        n = len(forced) if forced is not None else N_GREEDY
        for i in range(n):
            t = int(mx.argmax(y, -1).item())
            top_dec.append(t)
            nxt = int(forced[i]) if forced is not None else t
            toks.append(nxt)
            if forced is None and t in vl.eos:
                break
            y = vl.model(mx.array([[nxt]]), cache=cache)[0, -1:]
        vl.model.clear_mrope()
        out[f"{p['id']}_top_prefill"] = top_pre
        out[f"{p['id']}_top_decode"] = np.array(top_dec)
        out[f"{p['id']}_greedy"] = np.array(top_dec if forced is None else forced)
        ans = vl.decode_text(top_dec).strip().replace("\n", " ")
        if base is not None:
            ours = np.concatenate([top_pre, top_dec])
            theirs = np.concatenate([base[f"{p['id']}_top_prefill"], base[f"{p['id']}_top_decode"]])
            agree += int((ours == theirs).sum())
            total += len(ours)
            lines.append(f"P4 {a.name} {p['id']}: top-1 vs bf16 {100 * float((ours == theirs).mean()):.1f}% "
                         f"({len(ours)} pos) | {time.time() - t0:.1f}s | answer: {ans[:160]}")
        else:
            lines.append(f"P4 bf16 {p['id']}: {p['input_ids'].shape[1]} tok | {time.time() - t0:.1f}s | answer: {ans[:160]}")
        print(lines[-1], flush=True)
        mem(p["id"])
    np.savez(os.path.join(a.dir, f"p4-{a.name}.npz"), **out)
    if base is not None:
        print(f"P4 {a.name} SUMMARY: top-1 agreement vs bf16 {100 * agree / total:.2f}% over {total} positions "
              f"({len(load_inputs(a.dir))} image prompts)")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("p0"); p.add_argument("--hf", required=True); p.add_argument("--build", required=True)
    p.add_argument("--images", required=True); p.add_argument("--out", required=True)
    for name in ("p0-torch", "p1-ref", "p2", "p3-ref"):
        p = sp.add_parser(name); p.add_argument("--hf", required=True); p.add_argument("--dir", required=True)
        if name == "p3-ref":
            p.add_argument("--cases", default="shapes,two", help="comma list: shapes, two, text")
    p = sp.add_parser("text-inputs"); p.add_argument("--build", required=True); p.add_argument("--dir", required=True)
    p = sp.add_parser("p1-mlx"); p.add_argument("--hf", required=True); p.add_argument("--dir", required=True)
    p.add_argument("--vision", required=True)
    p = sp.add_parser("p3-mlx"); p.add_argument("--model"); p.add_argument("--dir", required=True)
    p.add_argument("--hf-weights", default=None, help="build the fp32 model from this HF folder (no bf16 rounding)")
    p.add_argument("--cases", default="shapes,two"); p.add_argument("--tol", type=float, default=1e-4)
    p = sp.add_parser("p4"); p.add_argument("--model", required=True); p.add_argument("--dir", required=True)
    p.add_argument("--name", required=True)
    a = ap.parse_args()
    {"p0": cmd_p0, "p0-torch": cmd_p0_torch, "p1-ref": cmd_p1_ref, "p1-mlx": cmd_p1_mlx, "p2": cmd_p2,
     "p3-ref": cmd_p3_ref, "p3-mlx": cmd_p3_mlx, "p4": cmd_p4, "text-inputs": cmd_text_inputs}[a.cmd](a)


if __name__ == "__main__":
    main()
