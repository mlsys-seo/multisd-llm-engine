# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Callable, List, Optional, Tuple, Union
import math
import torch
import torch.utils.checkpoint
from torch import nn
import flashinfer
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    logging,
)
from transformers.models.llama.configuration_llama import LlamaConfig
from ..request import FlashInferMetadata
from ..utils.common import profile_nvtx
from .ops.append_kv import append_paged_kv_cache

DEBUG = False
logger = logging.get_logger(__name__)


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaRotaryEmbedding(nn.Module):
    # def __init__(
    #     self,
    #     config: LlamaConfig,
    #     device=None,
    # ):
    def __init__(
        self,
        dim=None,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        rope_type="default",
        config: Optional[LlamaConfig] = None,
    ):
        super().__init__()
        self.rope_kwargs = {}
        if config is None:
            logger.warning_once(
                "`LlamaRotaryEmbedding` can now be fully parameterized by passing the model config through the "
                "`config` argument. All other arguments will be removed in v4.46"
            )
            self.rope_kwargs = {
                "rope_type": rope_type,
                "factor": scaling_factor,
                "dim": dim,
                "base": base,
                "max_position_embeddings": max_position_embeddings,
            }
            self.rope_type = rope_type
            self.max_seq_len_cached = max_position_embeddings
            self.original_max_seq_len = max_position_embeddings
        else:
            # BC: "rope_type" was originally "type"
            if config.rope_scaling is not None:
                self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
            else:
                self.rope_type = "default"
            self.max_seq_len_cached = config.max_position_embeddings
            self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, **self.rope_kwargs)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(
                self.config, device, seq_len=seq_len, **self.rope_kwargs
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: may break with compilation
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        inv_freq_expanded = self.inv_freq[:, None].float().expand(-1, 1)
        position_ids_expanded = position_ids[None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(0, 1)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

        self.is_gate_up_combined = False


    def combine_gate_up(self):
        hidden_size = self.gate_proj.weight.data.shape[1]
        intermediate_size = self.gate_proj.weight.data.shape[0]
        self.gate_up_proj = nn.Linear(hidden_size, intermediate_size + intermediate_size, bias=False, dtype=self.gate_proj.weight.dtype, device=self.gate_proj.weight.device)

        self.gate_up_proj.weight.data[:intermediate_size] = self.gate_proj.weight.data
        self.gate_up_proj.weight.data[intermediate_size:] = self.up_proj.weight.data
        
        if self.gate_up_proj.bias is not None:
            self.gate_up_proj.bias.data[:intermediate_size] = self.gate_proj.bias.data
            self.gate_up_proj.bias.data[intermediate_size:] = self.up_proj.bias.data

        del self.gate_proj
        del self.up_proj

        self.is_gate_up_combined = True

    def compute_gate_up(self, hidden_state):
        intermediate_size = self.config.intermediate_size
        if self.is_gate_up_combined:
            gate_slice, up_slice = torch.split(self.gate_up_proj(hidden_state), intermediate_size, dim=1)
            activated = self.act_fn(gate_slice)
            return activated * up_slice
        else:
            return self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state)

    def forward(self, hidden_state):
        activated = self.compute_gate_up(hidden_state)
        ret = self.down_proj(activated)
        return ret


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads


        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_key_value_heads * self.head_dim

        self.rotary_emb = LlamaRotaryEmbedding(
            self.head_dim, 
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = config.attention_dropout

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size * 2, self.q_size, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size * 2, self.kv_size, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size * 2, self.kv_size, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.q_size, self.hidden_size, bias=config.attention_bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        output_attentions: bool = False,
        # **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]
        
        if causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=0.0,
            is_causal=self.is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class LlamaFlashInferAttention(LlamaAttention):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.prefill_wrapper = config.prefill_wrapper
        self.decode_wrapper = config.decode_wrapper
        self.is_qkv_combined = False
        
    def combine_qkv(self):

        hidden_size = self.q_proj.weight.data.shape[1]
        q_size = self.q_proj.weight.data.shape[0]
        kv_size = self.k_proj.weight.data.shape[0]

        self.qkv_proj = nn.Linear(hidden_size, q_size + kv_size + kv_size, bias=True, dtype=self.q_proj.weight.dtype, device=self.q_proj.weight.device)
        self.qkv_proj.weight.data[:q_size, :] = self.q_proj.weight.data
        self.qkv_proj.weight.data[q_size:q_size + kv_size, :] = self.k_proj.weight.data
        self.qkv_proj.weight.data[q_size + kv_size:, :] = self.v_proj.weight.data
        self.qkv_proj.bias.data[:q_size] = False
        self.qkv_proj.bias.data[q_size:q_size + kv_size] = False
        self.qkv_proj.bias.data[q_size + kv_size:] = False

        self.q_size = q_size
        self.kv_size = kv_size
        del self.q_proj
        del self.k_proj
        del self.v_proj

        self.is_qkv_combined = True

    def compute_qkv(self, hidden_states):
        num_token = hidden_states.shape[0]

        if not self.is_qkv_combined:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
        else:
            qkv_states = self.qkv_proj(hidden_states)
            query_states, key_states, value_states = qkv_states.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        query_states = query_states.view(num_token, self.num_heads, self.head_dim).contiguous()
        key_states = key_states.view(num_token, self.num_key_value_heads, self.head_dim).contiguous()
        value_states = value_states.view(num_token, self.num_key_value_heads, self.head_dim).contiguous()
        return query_states, key_states, value_states
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        flashinfer_metadata: FlashInferMetadata = None,
        past_key_value: List[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        num_token = hidden_states.shape[0]
        query_states, key_states, value_states = self.compute_qkv(hidden_states)
        
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        append_paged_kv_cache(
            append_key=key_states,
            append_value=value_states,
            batch_indices=flashinfer_metadata.batch_indices,
            positions=flashinfer_metadata.positions,
            paged_kv_cache=past_key_value,
            kv_indices=self.config.prefill_wrapper._paged_kv_indices_buf,
            kv_indptr=self.config.prefill_wrapper._paged_kv_indptr_buf,
            kv_last_page_len=self.config.prefill_wrapper._paged_kv_last_page_len_buf,
            num_all_tokens=flashinfer_metadata.num_all_tokens_tensor
        )

        attn_output = self.config.prefill_wrapper.run(
            query_states,
            past_key_value,
        )

        attn_output = attn_output.reshape(num_token, self.hidden_size)

        if DEBUG:
            print(f"--------------- LlamaFlashInferAttention.forward ---------------")
            print(f"  key_states: {key_states.shape}")
            print(f"  value_states: {value_states.shape}")
            print(f"  query_states: {query_states.shape}")
            print(f"  attn_output: {attn_output.shape}")
            print(f"----------------------------------------------------------------")

        attn_output = self.o_proj(attn_output)
        return attn_output, None, None

class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int = 0):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaFlashInferAttention(config, layer_idx)
        self.mlp = LlamaMLP(config)
        self.hidden_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @profile_nvtx("LlamaDecoderLayer.forward")
    def forward(
        self,
        input_emb: torch.Tensor,
        hidden_states: torch.Tensor,
        flashinfer_metadata: FlashInferMetadata,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.hidden_norm(hidden_states)
        input_emb = self.input_layernorm(input_emb)

        hidden_states = torch.cat((input_emb, hidden_states), dim=-1)
        
        # Self Attention
        hidden_states, _, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            flashinfer_metadata=flashinfer_metadata,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Eagle3DraftLlamaForCausalLM(PreTrainedModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size

        # mlsys
        prefill_workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        self.prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            prefill_workspace_buffer, "NHD"
        )

        decode_workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        self.decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(    
            decode_workspace_buffer, "NHD"
        )
        setattr(config, "prefill_workspace_buffer", prefill_workspace_buffer)
        setattr(config, "decode_workspace_buffer", decode_workspace_buffer)
        setattr(config, "prefill_wrapper", self.prefill_wrapper)
        setattr(config, "decode_wrapper", self.decode_wrapper)
        self.q_data_type = config.torch_dtype
        self.kv_data_type = config.torch_dtype
        self.hidden_dtype = config.torch_dtype
        # mlsys

        self.rotary_emb = LlamaRotaryEmbedding(
            config.head_dim, 
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.lm_head = nn.Linear(config.hidden_size, config.draft_vocab_size, bias=False)
        hidden_size_multiplier = 3
        if hasattr(config, "target_hidden_size"):
            self.fc = nn.Linear(
                config.target_hidden_size * hidden_size_multiplier,
                self.hidden_size,
                bias=False,
            )
        else:
            self.fc = nn.Linear(
                config.hidden_size * hidden_size_multiplier,
                self.hidden_size,
                bias=False,
            )
        
        self.register_buffer(
            "d2t", torch.zeros(config.draft_vocab_size, dtype=torch.long)
        )
        self.register_buffer("t2d", torch.zeros(config.vocab_size, dtype=torch.bool))
        
        self.midlayer = LlamaDecoderLayer(config)
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.LongTensor = None,
        flashinfer_metadata: FlashInferMetadata = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = self.midlayer(
            input_emb=inputs_embeds,
            hidden_states=hidden_states,
            flashinfer_metadata=flashinfer_metadata,
            position_ids=position_ids,
            past_key_value=past_key_values[0],
        )
        return hidden_states

    def process_hidden_states(self, hidden_states):
        return self.fc(hidden_states.to(self.hidden_dtype))

    def get_logits(self, hidden_states):
        logits = self.lm_head(self.norm(hidden_states.to(self.hidden_dtype))).float()
        return logits
    
    def get_vocab_mapping(self, token_index):
        return self.d2t[token_index.to(self.d2t.device)]