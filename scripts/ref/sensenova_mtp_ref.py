# Vendored from OpenSenseNova/SenseNova-Loop src/sensenova_loop/models/mtp.py @ 840848460 (Apache-2.0), unchanged below.
"""Checkpoint-compatible Qwen3.5 MTP predictor for the Loop backbone.

The predictor consumes the preceding backbone state and the next token's
embedding. Its embedding and output projection are shared with the backbone.
The parameter layout follows the released ``mtp.*`` checkpoint tensors and
vLLM's Qwen3_5MultiTokenPredictor (Apache-2.0).
"""

import copy

import torch
from torch import nn
from transformers import DynamicCache
from transformers.models.qwen3_5 import modeling_qwen3_5 as hf


class SenseNovaLoopMTP(nn.Module):
    @staticmethod
    def configuration_error(config):
        """Describe unsupported optional predictors without blocking AR loading."""
        if getattr(config, "mtp_num_hidden_layers", 0) != 1:
            return "Loop MTP currently requires mtp_num_hidden_layers=1"
        if getattr(config, "mtp_use_dedicated_embeddings", False):
            return "Loop MTP requires shared embeddings and output weights"
        return None

    def __init__(self, config):
        super().__init__()
        self.config = copy.deepcopy(config)
        self.config.num_hidden_layers = 1
        self.config.layer_types = ["full_attention"]
        self.fc = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)
        self.pre_fc_norm_embedding = hf.Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_fc_norm_hidden = hf.Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layers = nn.ModuleList([hf.Qwen3_5DecoderLayer(self.config, 0)])
        self.norm = hf.Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = hf.Qwen3_5TextRotaryEmbedding(self.config)

    def forward(self, hidden_states, inputs_embeds, position_ids, *, past_key_values=None):
        if hidden_states.shape != inputs_embeds.shape:
            raise ValueError("MTP hidden states and next-token embeddings must have the same shape")
        cache = past_key_values if past_key_values is not None else DynamicCache(config=self.config)
        hidden = self.fc(torch.cat((self.pre_fc_norm_embedding(inputs_embeds),
                                    self.pre_fc_norm_hidden(hidden_states)), dim=-1))
        mask = hf.create_causal_mask(config=self.config, inputs_embeds=hidden,
                                     attention_mask=None, past_key_values=cache)
        rotary = self.rotary_emb(hidden, position_ids)
        hidden = self.layers[0](hidden, attention_mask=mask, position_embeddings=rotary,
                                past_key_values=cache)
        return self.norm(hidden), cache
