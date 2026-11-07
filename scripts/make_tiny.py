#!/usr/bin/env python3
"""Make a tiny random-weight SenseNova-Loop checkpoint (HF layout, float32) plus reference logits.

  --ref ours    : PyTorch re-implementation (scripts/torch_loop_ref.py) on Transformers' Qwen3.5 layers  (cloud)
  --ref remote  : SenseNova's own modeling_sensenova_loop.py from --src (the downloaded model folder)   (Mac, CPU)

Layout of the tiny model: 8 layers (full attention at 3 and 7), loop [2, 6) x2, two-stream mHC on [2, 6), so the plan
is 0 1 | 2 3 4 5 | 2 3 4 5 | 6 7 (12 visits) and the loop holds both a linear and a full-attention layer.
The mHC router weights are random (std 0.05) so routing depends on the input; zero weights would hide bugs.
"""
import argparse, json, os, shutil, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

TEXT = dict(model_type="qwen3_5_text", hidden_size=64, intermediate_size=128, num_hidden_layers=8,
            num_attention_heads=4, num_key_value_heads=2, head_dim=16, linear_num_key_heads=2,
            linear_num_value_heads=4, linear_key_head_dim=16, linear_value_head_dim=16, linear_conv_kernel_dim=4,
            vocab_size=512, max_position_embeddings=4096, rms_norm_eps=1e-6, full_attention_interval=4,
            attn_output_gate=True, attention_bias=False, tie_word_embeddings=False, hidden_act="silu",
            rope_parameters=dict(rope_type="default", rope_theta=10000000, partial_rotary_factor=0.25,
                                 mrope_section=[1, 1, 0], mrope_interleaved=True),
            partial_rotary_factor=0.25, pad_token_id=0, eos_token_id=1, mlp_only_layers=[], dtype="float32")
TEXT["layer_types"] = ["full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(8)]
LOOP = dict(start=2, end=6, eval_count=2, counts=[2])
MHC = dict(start=2, end=6, streams=2, iterations=20, pre_scale=1.0, fixed_identity=False)
VISION = dict(model_type="qwen3_5_vision", depth=1, hidden_size=32, intermediate_size=64, num_heads=2, in_channels=3,
              patch_size=16, spatial_merge_size=2, temporal_patch_size=2, out_hidden_size=64,
              num_position_embeddings=64, hidden_act="gelu_pytorch_tanh", deepstack_visual_indexes=[])


def mhc_bias(s):
    p = torch.tensor({2: (0.25, 0.75), 4: (0.0625, 0.1875, 0.25, 0.5)}[s])
    diag = float(np.log((1 - (s - 1) / 16) * 16))
    return torch.cat([torch.logit(p), torch.zeros(s), (torch.eye(s) * diag).flatten()])


def write_tokenizer(out, vocab):
    """Throw-away word-level tokenizer so mlx_lm.convert / load work on the tiny folder."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    tok = Tokenizer(models.WordLevel({f"t{i}": i for i in range(vocab)}, unk_token="t0"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.save(os.path.join(out, "tokenizer.json"))
    json.dump({"tokenizer_class": "PreTrainedTokenizerFast", "eos_token": "t1", "pad_token": "t0"},
              open(os.path.join(out, "tokenizer_config.json"), "w"))


def cases(n, L, vocab, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(2, vocab, (1, L), generator=g) for _ in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref", choices=["ours", "remote"], default="ours")
    ap.add_argument("--src", help="downloaded model folder with modeling_sensenova_loop.py (for --ref remote)")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--len", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-loop", action="store_true", help="eval_count 1 (for the loop-fires check)")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    loop = dict(LOOP)
    if a.no_loop:
        loop.update(eval_count=1, counts=[1])
    cfg = dict(architectures=["SenseNovaLoopForConditionalGeneration"], model_type="sensenova_loop",
               text_config=TEXT, vision_config=VISION, loop_settings=loop, mhc=MHC, tie_word_embeddings=False,
               hidden_size=64, image_token_id=500, video_token_id=501, vision_start_token_id=502,
               vision_end_token_id=503, dtype="float32")
    from safetensors.torch import save_file
    ids = cases(a.n, a.len, TEXT["vocab_size"], a.seed + 1)

    if a.ref == "ours":
        from transformers import Qwen3_5TextConfig
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel
        import torch_loop_ref as R
        tm = Qwen3_5TextModel(Qwen3_5TextConfig(**{k: v for k, v in TEXT.items() if k != "model_type"})).float().eval()
        # random norm offsets so the (1 + w) convention is exercised
        for n_, p in tm.named_parameters():
            if n_.endswith("layernorm.weight") or n_ == "norm.weight" or n_.endswith("q_norm.weight") or n_.endswith("k_norm.weight"):
                p.data.normal_(0, 0.1)
        lm_head = torch.nn.Linear(64, TEXT["vocab_size"], bias=False)
        mp = {}
        for i in range(MHC["start"], MHC["end"]):
            mp[i] = {k: (torch.randn(8, 128) * 0.05, mhc_bias(2) + torch.randn(8) * 0.1) for k in ("attn", "ffn")}
        sd = {f"model.language_model.{k}": v.contiguous() for k, v in tm.state_dict().items()}
        sd["lm_head.weight"] = lm_head.weight.data.contiguous()
        for i, d in mp.items():
            for k, (w, b) in d.items():
                sd[f"model.language_model.layers.{i}.mhc_{k}.weight"] = w.contiguous()
                sd[f"model.language_model.layers.{i}.mhc_{k}.bias"] = b.contiguous()
        save_file(sd, os.path.join(a.out, "model.safetensors"))
        json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=1)
        logits = [R.forward(tm, lm_head, mp, x, loop, MHC).float().numpy() for x in ids]
    else:
        from transformers import AutoConfig, AutoModelForCausalLM
        shutil.copy(os.path.join(a.src, "modeling_sensenova_loop.py"), a.out)
        real = json.load(open(os.path.join(a.src, "config.json")))
        cfg["auto_map"] = real["auto_map"]
        json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=1)
        c = AutoConfig.from_pretrained(a.out, trust_remote_code=True)
        m = AutoModelForCausalLM.from_config(c, trust_remote_code=True).float().eval()
        with torch.no_grad():
            for n_, p in m.named_parameters():
                if ".mhc_" in n_ and n_.endswith("weight"):
                    p.normal_(0, 0.05)
                elif ".mhc_" in n_ and n_.endswith("bias"):
                    p.copy_(mhc_bias(2) + torch.randn(8) * 0.1)
                elif "language_model" in n_ and (n_.endswith("layernorm.weight") or n_.endswith(".norm.weight") or "q_norm" in n_ or "k_norm" in n_):
                    p.normal_(0, 0.1)
        n_mhc = sum(".mhc_" in k for k in m.state_dict())
        assert n_mhc == 4 * (MHC["end"] - MHC["start"]), f"expected mHC params in the remote model, got {n_mhc}"
        m.save_pretrained(a.out, safe_serialization=True)
        with torch.no_grad():
            logits = [m(input_ids=x, use_cache=False).logits.float().numpy() for x in ids]
    write_tokenizer(a.out, TEXT["vocab_size"])
    np.savez(os.path.join(a.out, "ref_logits.npz"), ids=np.stack([x.numpy()[0] for x in ids]), logits=np.stack([l[0] for l in logits]))
    print(f"tiny model ({a.ref}) -> {a.out}: {a.n} cases x {a.len} tokens")


if __name__ == "__main__":
    main()
