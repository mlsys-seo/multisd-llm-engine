import os
import time
import json
from safetensors import safe_open

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

import ray
import torch
import flashinfer
import numpy as np

from .registry import EngineRegistry
from .request import FlashInferMetadata
from .sampler import Sampler, SamplingParams
from .utils.parallel_group_utils import (
    ParallelStatus,
    RunnerParallelGroup,
    ExecutorParallelGroup,
    get_tensor_ipc_info,
    rebuild_tensor_from_ipc_info
)
from .utils.parallel_utils import load_or_create_tp_model
from .utils.common import profile_nvtx, print_debug

DEBUG = False

@dataclass
class CUDAGraphCache:
    batch_size: int
    num_tokens: int
    prefill_wrapper: flashinfer.BatchPrefillWithPagedKVCacheWrapper = None
    graph: torch.cuda.CUDAGraph = None
    input_tensors: Tuple[torch.Tensor] = None
    output_tensor: torch.Tensor = None
    flashinfer_metadata: FlashInferMetadata = None


class ModelExecutor:
    def __init__(
        self,
        registry: EngineRegistry,
        type = None,
        parallel_status: Optional[ParallelStatus] = None,
        **kwargs
    ):
        if type is None:
            model_type = "model"
        else:
            model_type = f"{type}_model"

        self.model_type = model_type
        self.registry = registry
        
        self.use_cuda_graph = self.registry.get('val.engine.use_cuda_graph')
        self.vocab_size = self.registry.get(f'val.engine.vocab_size')
        self.model_name_or_path = self.registry.get(f'val.{model_type}.model_name_or_path')
        self.model_cache_dir = self.registry.get(f'val.engine.model_cache_dir')
        self.model_cls = self.registry.get(f'val.{model_type}.model_class')
        self.hf_config = self.registry.get(f'val.{model_type}.hf_config')
        self.dtype = self.hf_config.torch_dtype if self.hf_config.torch_dtype != torch.float32 else torch.float16
        self.past_key_values_dtype = self.hf_config.torch_dtype if self.hf_config.torch_dtype != torch.float32 else torch.float16
        
        self.local_rank = parallel_status.local_rank
        self.world_size = len(parallel_status.device_ids)

        self.use_ray_worker = self.registry.get('val.engine.use_ray_worker')
        self.use_ray_executor = self.registry.get('val.engine.use_ray_executor')

        if self.use_ray_worker or self.use_ray_executor:
            self.device = 'cuda:0'
        else:
            self.device = 'cuda:0'


        self.use_data_parallel_draft = self.registry.get('val.engine.use_data_parallel_draft')
        if self.use_data_parallel_draft and self.model_type != 'model':
            self.num_kv_cache_blocks = self.registry.get('val.engine.num_kv_cache_blocks') // self.world_size
        else:
            self.num_kv_cache_blocks = self.registry.get('val.engine.num_kv_cache_blocks')
        self.kv_cache_block_size = self.registry.get('val.engine.kv_cache_block_size')
        
        setattr(self.hf_config, 'num_draft_steps', self.registry.get('val.engine.num_draft_steps'))
        setattr(self.hf_config, 'max_batch_size', self.registry.get('val.engine.num_max_batch_requests'))
        setattr(self.hf_config, 'kv_cache_block_size', self.kv_cache_block_size)
        setattr(self.hf_config, 'past_key_values_dtype', self.past_key_values_dtype)
        setattr(self.hf_config, 'torch_dtype', self.dtype)
        setattr(self.hf_config, 'past_key_values_dtype', self.past_key_values_dtype)

        self.captured_models = defaultdict(dict)
        self.graph_target_batch_sizes = self.registry.get('val.engine.cuda_graph_target_batch_sizes')
        
        self.quantization_type = self.registry.get('val.engine.quantization')

        engine_has_speculation = self.registry.get('val.draft_model.model_name_or_path') is not None
        engine_has_hierarchical_speculation = self.registry.get('val.verify_model.model_name_or_path') is not None

        self.required_all_gather = (engine_has_speculation or engine_has_hierarchical_speculation) and self.use_data_parallel_draft and self.model_type == 'model'

        self.capture_only_attn = False
        if type is None:
            if engine_has_hierarchical_speculation:
                self.graph_target_num_tokens = [self.registry.get('val.engine.num_draft_steps') + 1]
                self.graph_target_batch_sizes = []
                # self.capture_only_attn = True
                
            elif engine_has_speculation:
                self.graph_target_num_tokens = [self.registry.get('val.engine.num_draft_steps'), self.registry.get('val.engine.num_draft_steps') + 1]
                self.graph_target_batch_sizes = []
                # self.capture_only_attn = True
            else:
                self.graph_target_num_tokens = [1]
                self.graph_target_batch_sizes = []
                # self.capture_only_attn = True
        elif type == "draft":
            if engine_has_hierarchical_speculation:
                self.graph_target_num_tokens = [1, 2]
            else:
                self.graph_target_num_tokens = [1, 2]
        
        elif type == "verify":
            self.graph_target_num_tokens = [self.registry.get('val.engine.num_draft_steps'), self.registry.get('val.engine.num_draft_steps') + 1]

        self.eagle3_enabled = self.registry.get('val.engine.eagle3_enabled')
        
        self.past_key_values = None

        if self.model_type != 'model':
            self.use_cache_offloading = self.registry.get('val.engine.use_cache_offloading')
        else:
            self.use_cache_offloading = False

        self.model = self.load_model()

        if self.registry.get('val.engine.use_cuda_graph'):
            self.build_graph()

        self.use_sparse_probs = self.registry.get('val.engine.use_sparse_probs')
        self.sampler = Sampler(self.vocab_size).eval()
        

    def get_generation_config(self):
        return self.model.generation_config


    def build_key_value_cache_blocks(self):
        # initialize model cache
        num_layers = self.hf_config.num_hidden_layers
        num_kv_heads = self.hf_config.num_key_value_heads
        if "qwen3" in self.hf_config.name_or_path.lower():
            head_dim = self.hf_config.head_dim
        else:
            head_dim = self.hf_config.hidden_size // self.hf_config.num_attention_heads
        self.past_key_values = [
            torch.randn(
                self.num_kv_cache_blocks, 
                2, 
                self.kv_cache_block_size, 
                num_kv_heads, 
                head_dim, 
                dtype=self.past_key_values_dtype, 
                device=self.device
            ) for _ in range(num_layers)
        ]

        if self.use_cache_offloading:
            self.cpu_cache = [
                torch.randn(
                    *self.hf_config.past_key_values_metadata['shape'],
                    dtype=self.hf_config.past_key_values_metadata['dtype'], 
                    device='cpu',
                    pin_memory=True
                ) for _ in range(num_layers)
            ]
        else:
            self.cpu_cache = [None for _ in range(num_layers)]
        print(f"Paged KV block pool: {self.num_kv_cache_blocks} blocks were initialized")


    def _build_flashinfer_graph_buffers(self, batch_size):
        max_kv_indices_len_per_request = self.hf_config.max_position_embeddings // self.hf_config.kv_cache_block_size
        max_kv_indices_len = max_kv_indices_len_per_request * batch_size
        q_indptr_buffer = torch.empty(batch_size + 1, device=self.device).int()
        kv_indptr_buffer = torch.empty(batch_size + 1, device=self.device).int()
        kv_indices_buffer = torch.empty(max_kv_indices_len, device=self.device).int()
        kv_last_page_len_buffer = torch.empty(batch_size, device=self.device).int()
        
        return q_indptr_buffer, kv_indptr_buffer, kv_indices_buffer, kv_last_page_len_buffer
    

    def build_prefill_wrapper(self, use_cuda_graph=False, batch_size=None):
        if not use_cuda_graph and batch_size is None:
            prefill_workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=self.device)
            prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                prefill_workspace_buffer, "NHD", False
            )
        else:
            graph_buffers = self._build_flashinfer_graph_buffers(batch_size)
            prefill_workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=self.device)
            prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                prefill_workspace_buffer, "NHD", True, *graph_buffers
            )

        return prefill_wrapper

    
    def load_model(self):

        self.non_graph_prefill_wrapper = self.build_prefill_wrapper(use_cuda_graph=False)
        
        self.non_graph_wrappers = {}
        for batch_size in self.graph_target_batch_sizes:
            self.non_graph_wrappers[batch_size] = self.build_prefill_wrapper(use_cuda_graph=False, batch_size=batch_size)

        for batch_size in self.graph_target_batch_sizes:
            for num_tokens in self.graph_target_num_tokens:
                self.captured_models[batch_size][num_tokens] = CUDAGraphCache(
                    batch_size=batch_size,
                    num_tokens=num_tokens,
                    prefill_wrapper=self.build_prefill_wrapper(use_cuda_graph=True, batch_size=batch_size)
                )
        setattr(self.hf_config, 'use_cache_offloading', self.registry.get('val.engine.use_cache_offloading'))
        if self.registry.get('val.engine.use_cache_offloading') and self.model_type != 'model':
            shape = (
                self.num_kv_cache_blocks, 
                2, 
                self.kv_cache_block_size, 
                self.hf_config.num_key_value_heads, 
                self.hf_config.hidden_size // self.hf_config.num_attention_heads)

            dtype = self.past_key_values_dtype
            setattr(self.hf_config, 'past_key_values_metadata', {'shape': shape, 'dtype': dtype})

        if self.eagle3_enabled and self.model_type == "draft_model":
            model = self.load_eagle3_model(self.model_name_or_path)
        else:
            try:
                model = self.model_cls.from_pretrained(self.model_name_or_path, torch_dtype=self.dtype, config=self.hf_config, cache_dir=self.model_cache_dir, use_safetensors=True, token=self.registry.get('val.engine.hf_model_token'))
            except Exception as e:
                model = self.model_cls.from_pretrained(self.model_name_or_path, torch_dtype=self.dtype, config=self.hf_config, cache_dir=self.model_cache_dir, use_safetensors=False, token=self.registry.get('val.engine.hf_model_token'))

        model.eval()
        model.to(self.device)
        # if self.quantization_type is None:
        #     model.combine_qkv()

        self.build_key_value_cache_blocks() # lazy init
        
        if DEBUG:
            print(f"Model {self.model_name_or_path} is loaded on device {self.device}")
        return model

    def load_eagle3_model(self, model_name_or_path):
        # model = self.model_cls(config=self.hf_config)
        model = self.model_cls.from_pretrained(model_name_or_path, torch_dtype=self.dtype, config=self.hf_config, cache_dir=self.model_cache_dir, use_safetensors=False, token=self.registry.get('val.engine.hf_model_token'))
        if "AngelSlim/Qwen3-32B_eagle3" in model_name_or_path:
            # pretrained_model = torch.load(os.path.join(self.model_cache_dir, "models--AngelSlim--Qwen3-32B_eagle3/pytorch_model.bin"), map_location="cpu")
            # model.load_state_dict(pretrained_model, strict=False)
            embedding_json = json.load(open(os.path.join(self.model_cache_dir, "models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137/model.safetensors.index.json")))
            embedding_path = os.path.join(os.path.join(self.model_cache_dir, "models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137/"), embedding_json["weight_map"]["model.embed_tokens.weight"])
        if "AngelSlim/Qwen3-14B_eagle3" in model_name_or_path:
            # pretrained_model = torch.load(os.path.join(self.model_cache_dir, "models--AngelSlim--Qwen3-14B_eagle3/pytorch_model.bin"), map_location="cpu")
            # model.load_state_dict(pretrained_model, strict=False)
            embedding_json = json.load(open(os.path.join(self.model_cache_dir, "models--Qwen--Qwen3-14B/snapshots/40c069824f4251a91eefaf281ebe4c544efd3e18/model.safetensors.index.json")))
            embedding_path = os.path.join(os.path.join(self.model_cache_dir, "models--Qwen--Qwen3-14B/snapshots/40c069824f4251a91eefaf281ebe4c544efd3e18/"), embedding_json["weight_map"]["model.embed_tokens.weight"])
        elif "AngelSlim/Qwen3-8B_eagle3" in model_name_or_path:
            # pretrained_model = torch.load(os.path.join(self.model_cache_dir, "models--AngelSlim--Qwen3-8B_eagle3/pytorch_model.bin"), map_location="cpu")
            # model.load_state_dict(pretrained_model, strict=False)
            embedding_json = json.load(open(os.path.join(self.model_cache_dir, "models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218/model.safetensors.index.json")))
            embedding_path = os.path.join(os.path.join(self.model_cache_dir, "models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218/"), embedding_json["weight_map"]["model.embed_tokens.weight"])
        elif "AngelSlim/Qwen3-4B_eagle3" in model_name_or_path:
            # pretrained_model = torch.load(os.path.join(self.model_cache_dir, "models--AngelSlim--Qwen3-4B_eagle3/pytorch_model.bin"), map_location="cpu")
            # model.load_state_dict(pretrained_model, strict=False)
            embedding_json = json.load(open(os.path.join(self.model_cache_dir, "models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c/model.safetensors.index.json")))
            embedding_path = os.path.join(os.path.join(self.model_cache_dir, "models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c/"), embedding_json["weight_map"]["model.embed_tokens.weight"])
        elif "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B" in model_name_or_path:
            # pretrained_model = torch.load(os.path.join(self.model_cache_dir, "models--yuhuili--EAGLE3-LLaMA3.1-Instruct-8B/pytorch_model.bin"), map_location="cpu")
            # model.load_state_dict(pretrained_model, strict=False)
            embedding_json = json.load(open(os.path.join(self.model_cache_dir, "models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659/model.safetensors.index.json")))
            embedding_path = os.path.join(os.path.join(self.model_cache_dir, "models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659/"), embedding_json["weight_map"]["model.embed_tokens.weight"])
            
        with safe_open(embedding_path, framework="pt") as f:
            tensor_slice = f.get_slice("model.embed_tokens.weight")
            _, hidden_dim = tensor_slice.get_shape()
            embeddings_weight = tensor_slice[:, :hidden_dim].float()
        model.embed_tokens.weight.data = embeddings_weight

        model.to(self.dtype)
        return model
    
    def _build_flashinfer_dummy_inputs(self, batch_size, num_tokens):
        input_ids_list = []
        paged_kv_indices_list = []
        paged_kv_last_page_len_list = []
        block_idx = 0
        for i in range(batch_size):
            input_ids = np.arange(i*num_tokens, (i+1)*num_tokens)
            num_blocks = np.ceil(num_tokens / self.hf_config.kv_cache_block_size).astype(int)
            paged_kv_indices = [block_idx + i for i in range(num_blocks)]
            paged_kv_last_page_len = np.array([num_tokens % self.hf_config.kv_cache_block_size])
            block_idx += num_blocks

            input_ids_list.append(input_ids)
            paged_kv_indices_list.append(paged_kv_indices)
            paged_kv_last_page_len_list.append(paged_kv_last_page_len)

        
        input_ids_nested = np.concatenate(input_ids_list, axis=0)
        input_ids_indptr = np.cumsum([0] + [len(input_ids) for input_ids in input_ids_list])
        input_ids_lengths = [len(input_ids) for input_ids in input_ids_list]
        paged_kv_indptr = np.cumsum([0] + [len(paged_kv_indices) for paged_kv_indices in paged_kv_indices_list])
        paged_kv_indices = np.concatenate(paged_kv_indices_list, axis=0)

        paged_kv_indices_len = paged_kv_indices.shape[0]

        max_kv_indices_len_per_request = self.hf_config.max_position_embeddings // self.hf_config.kv_cache_block_size
        max_kv_indices_len = max_kv_indices_len_per_request * batch_size
        paged_kv_indices = np.pad(paged_kv_indices, (0, max_kv_indices_len - paged_kv_indices_len), mode='constant', constant_values=0)

        paged_kv_last_page_len = np.concatenate(paged_kv_last_page_len_list, axis=0)

        input_ids_tensor = torch.tensor(input_ids_nested, dtype=torch.long)
        input_ids_indptr = torch.tensor(input_ids_indptr, dtype=torch.int32)
        input_ids_lengths = torch.tensor(input_ids_lengths, dtype=torch.int32)
        paged_kv_indices = torch.tensor(paged_kv_indices, dtype=torch.int32)
        paged_kv_indptr = torch.tensor(paged_kv_indptr, dtype=torch.int32)
        paged_kv_last_page_len = torch.tensor(paged_kv_last_page_len, dtype=torch.int32)
        
        # batch_indices, positions = flashinfer.get_batch_indices_positions(
        #     input_ids_indptr,
        #     flashinfer.get_seq_lens(paged_kv_indptr, paged_kv_last_page_len, self.kv_cache_block_size),
        #     input_ids_tensor.shape[0]
        # )
        return FlashInferMetadata(
            input_ids_indptr=input_ids_indptr,
            input_ids_lengths=input_ids_lengths,
            batch_indices=None,
            positions=None,
            paged_kv_indptr=paged_kv_indptr,
            paged_kv_indices=paged_kv_indices,
            paged_kv_last_page_len=paged_kv_last_page_len,
            kv_cache_block_size=self.kv_cache_block_size
        )
    

    def _build_dummy_inputs(self, batch_size, num_tokens):
        input_ids_tensor = torch.randint(0, 100, (batch_size * num_tokens,), dtype=torch.int64, device=self.device)
        position_ids_tensor = torch.arange(0, batch_size * num_tokens, dtype=torch.int64, device=self.device)

        if self.eagle3_enabled and self.model_type == "draft_model":
            hidden_states_tensor = torch.randn((batch_size * num_tokens, self.hf_config.hidden_size), dtype=self.hf_config.torch_dtype, device=self.device)
            return input_ids_tensor, position_ids_tensor, hidden_states_tensor
        # print(f"\n============== _build_dummy_inputs batch_size: {batch_size} num_tokens: {num_tokens} ==============")
        # print(f"input_ids_tensor: {input_ids_tensor.shape}")
        # print(f"position_ids_tensor: {position_ids_tensor.shape}")
        return input_ids_tensor, position_ids_tensor, None


    def build_graph(self):
        for batch_size in self.graph_target_batch_sizes:
            for num_tokens in self.graph_target_num_tokens:
                print(f"Building graph for batch size {batch_size} num_tokens {num_tokens}")
                input_tensors = self._build_dummy_inputs(batch_size, num_tokens)

                flashinfer_metadata = self._build_flashinfer_dummy_inputs(batch_size, num_tokens)
                self.captured_models[batch_size][num_tokens].prefill_wrapper.plan(
                    qo_indptr=flashinfer_metadata.input_ids_indptr,
                    paged_kv_indptr=flashinfer_metadata.paged_kv_indptr,
                    paged_kv_indices=flashinfer_metadata.paged_kv_indices,
                    paged_kv_last_page_len=flashinfer_metadata.paged_kv_last_page_len,
                    num_qo_heads=self.hf_config.num_attention_heads,
                    num_kv_heads=self.hf_config.num_key_value_heads,
                    head_dim_qk=self.hf_config.hidden_size // self.hf_config.num_attention_heads if not "qwen3" in self.hf_config.name_or_path.lower() else self.hf_config.head_dim,
                    page_size=self.hf_config.kv_cache_block_size,
                    causal=True,
                    pos_encoding_mode="NONE", # TODO: Check
                    use_fp16_qk_reduction=False,
                    q_data_type=self.hf_config.torch_dtype,
                    kv_data_type=self.past_key_values_dtype,
                )
            
            
                self.model.config.prefill_wrapper = self.captured_models[batch_size][num_tokens].prefill_wrapper

                flashinfer_metadata.set_append_metadata(
                    prefill_wrapper=self.captured_models[batch_size][num_tokens].prefill_wrapper,
                    kv_cache_block_size=self.hf_config.kv_cache_block_size,
                )

                # warmup
                for i in range(10):
                    output_tensor = self._run_model_eager(*input_tensors, flashinfer_metadata=flashinfer_metadata)
                
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    output_tensor = self._run_model_eager(*input_tensors, flashinfer_metadata=flashinfer_metadata)
                torch.cuda.synchronize()

                self.captured_models[batch_size][num_tokens].graph = g
                self.captured_models[batch_size][num_tokens].input_tensors = input_tensors
                self.captured_models[batch_size][num_tokens].output_tensor = output_tensor
                self.captured_models[batch_size][num_tokens].flashinfer_metadata = flashinfer_metadata
    

    def is_able_to_run_graph(
            self,
            num_all_tokens: int,
            batch_size: int
        ):
        if not self.use_cuda_graph:
            return False

        if not self.graph_target_batch_sizes:
            return False

        # if batch_size > max(self.graph_target_batch_sizes):
        #     return False

        if self.capture_only_attn:
            return False
        
        if batch_size not in self.graph_target_batch_sizes:
            return False

        if num_all_tokens > self.graph_target_num_tokens[-1] * batch_size:
            return False

        return True
    
    def onloading_cache(self):
        for decoder_layer_idx in range(len(self.past_key_values)):
            self.past_key_values[decoder_layer_idx].copy_(self.cpu_cache[decoder_layer_idx], non_blocking=True)

    def offloading_cache(self):
        for decoder_layer_idx in range(len(self.past_key_values)):
            self.cpu_cache[decoder_layer_idx].copy_(self.past_key_values[decoder_layer_idx], non_blocking=True)
    
    @profile_nvtx("ModelExecutor.run_model")
    def _run_model(
            self, 
            input_ids_tensor: torch.Tensor = None,
            position_ids_tensor: torch.Tensor = None,
            flashinfer_metadata: FlashInferMetadata = None,
            sampling_params: List[SamplingParams] = None,
            num_logits_to_keep: int = None,
            hidden_states: Optional[torch.Tensor] = None,
            **kwargs
        ):

        num_all_tokens = input_ids_tensor.shape[0]
        batch_size = flashinfer_metadata.input_ids_indptr_cpu.shape[0] - 1

        if self.is_able_to_run_graph(num_all_tokens, batch_size):
            hidden_states = self._run_model_graph(
                batch_size=batch_size,
                input_ids_tensor=input_ids_tensor,
                position_ids_tensor=position_ids_tensor,
                flashinfer_metadata=flashinfer_metadata,
                hidden_states=hidden_states,
            )
        
        else:
            input_ids_tensor = input_ids_tensor.to(self.device, non_blocking=True)
            position_ids_tensor = position_ids_tensor.to(self.device, non_blocking=True)
            if self.eagle3_enabled and hidden_states is not None:
                hidden_states = hidden_states.to(self.device, non_blocking=True)
            
            self.non_graph_prefill_wrapper.plan(
                qo_indptr=flashinfer_metadata.input_ids_indptr,
                paged_kv_indptr=flashinfer_metadata.paged_kv_indptr,
                paged_kv_indices=flashinfer_metadata.paged_kv_indices,
                paged_kv_last_page_len=flashinfer_metadata.paged_kv_last_page_len,
                num_qo_heads=self.hf_config.num_attention_heads,
                num_kv_heads=self.hf_config.num_key_value_heads,
                head_dim_qk=self.hf_config.hidden_size // self.hf_config.num_attention_heads if not "qwen3" in self.hf_config.name_or_path.lower() else self.hf_config.head_dim,
                page_size=self.hf_config.kv_cache_block_size,
                causal=True,
                pos_encoding_mode="NONE", # TODO: Check
                use_fp16_qk_reduction=False,
                q_data_type=self.hf_config.torch_dtype,
                kv_data_type=self.past_key_values_dtype,
            )
            self.model.config.prefill_wrapper = self.non_graph_prefill_wrapper
            flashinfer_metadata.set_append_metadata(
                prefill_wrapper=self.non_graph_prefill_wrapper,
                kv_cache_block_size=self.hf_config.kv_cache_block_size,
            )
            
            hidden_states = self._run_model_eager(
                input_ids_tensor=input_ids_tensor,
                position_ids_tensor=position_ids_tensor,
                flashinfer_metadata=flashinfer_metadata,
                hidden_states=hidden_states,
                **kwargs
            )
        
        return hidden_states

    def get_logits(self, hidden_states):
        logits = self.model.get_logits(hidden_states)
        return logits
    
    def run_model(
            self,
            input_ids_tensor: torch.Tensor,
            position_ids_tensor: torch.Tensor,
            flashinfer_metadata: FlashInferMetadata,
            sampling_params: List[SamplingParams] = None,
            num_logits_to_keep: int = None,
            hidden_states: Optional[torch.Tensor] = None,
            **kwargs
        ):

        if self.eagle3_enabled and self.model_type == "draft_model" and hidden_states.shape[-1] != self.hf_config.hidden_size:
            hidden_states = self.model.process_hidden_states(hidden_states)
        
        outputs = self._run_model(
            input_ids_tensor=input_ids_tensor,
            position_ids_tensor=position_ids_tensor,
            flashinfer_metadata=flashinfer_metadata,
            sampling_params=sampling_params,
            num_logits_to_keep=num_logits_to_keep,
            hidden_states=hidden_states,
            **kwargs
        )
        
        # jkim : for eagle3 support 
        if self.eagle3_enabled and type(outputs) is tuple:
            hidden_states = outputs[0]
        else:
            hidden_states = outputs
        
        start_layer_idx = kwargs.get('start_layer_idx', None)
        end_layer_idx = kwargs.get('end_layer_idx', None)
        
        probs_input_ids_indptr = flashinfer_metadata.input_ids_indptr_cpu
        input_ids_lengths = flashinfer_metadata.input_ids_lengths.numpy().astype(np.int32)
        cuda_indices = None  # Initialize cuda_indices
        if num_logits_to_keep is not None:
            if num_logits_to_keep == 0:
                return [], [], None
            
            if num_logits_to_keep == 1:
                slice_indices = probs_input_ids_indptr[1:] - 1
                cuda_indices = torch.from_numpy(slice_indices).to(hidden_states.device)

                hidden_states = hidden_states.index_select(0, cuda_indices)
                probs_input_ids_indptr = np.arange(flashinfer_metadata.input_ids_indptr.shape[0])
                input_ids_lengths = np.ones(input_ids_lengths.shape[0], dtype=np.int32)
            else:
                if end_layer_idx is not None and  start_layer_idx < end_layer_idx:
                    end_indices = flashinfer_metadata.input_ids_indptr_cpu[1:]-1
                else:
                    end_indices = flashinfer_metadata.input_ids_indptr_cpu[1:]
                num_tokens = np.minimum(input_ids_lengths, num_logits_to_keep)
                start_indices = end_indices - num_tokens
                slice_indices = np.concatenate([np.arange(s, e) for s, e in zip(start_indices, end_indices)])
                cuda_indices = torch.from_numpy(slice_indices).to(hidden_states.device)
                hidden_states = hidden_states.index_select(0, cuda_indices)
                probs_input_ids_indptr = np.concatenate([[0], np.cumsum(num_tokens)])
                input_ids_lengths = num_tokens

        logits = self.get_logits(hidden_states)
        
        sampled_tokens_per_request, probs = self.run_sampler(
            logits,
            input_ids_indptr=probs_input_ids_indptr,
            sampling_params=sampling_params,
            return_log_probs=False
        )

        if self.eagle3_enabled:
            if type(outputs) is tuple:
                hidden_states = torch.cat(outputs[1], dim=-1)
            hidden_states_list = torch.split(hidden_states, input_ids_lengths.tolist(), dim=0)
            return sampled_tokens_per_request, probs, probs_input_ids_indptr, hidden_states_list

        # if self.registry.get('val.engine.use_sparse_probs'):
        #     probs_per_request = [probs.to_sparse() for probs in probs_per_request]
        return sampled_tokens_per_request, probs, probs_input_ids_indptr
 
    @profile_nvtx("ModelExecutor._run_model_eager")
    @torch.no_grad()
    def _run_model_eager(self, input_ids_tensor, position_ids_tensor, hidden_states, flashinfer_metadata, **kwargs):
        
        hidden_states = self.model(
            input_ids=input_ids_tensor,
            position_ids=position_ids_tensor,
            flashinfer_metadata=flashinfer_metadata,
            past_key_values=self.past_key_values,
            hidden_states=hidden_states,
            **kwargs
        )
        return hidden_states

    @profile_nvtx("ModelExecutor._run_model_graph")
    @torch.no_grad()
    def _run_model_graph(
            self,
            batch_size,
            input_ids_tensor: torch.Tensor = None,
            position_ids_tensor: torch.Tensor = None,
            flashinfer_metadata: FlashInferMetadata = None,
            hidden_states: Optional[torch.Tensor] = None,
        ):
        
        candidates = [num for num in self.graph_target_num_tokens if num >= input_ids_tensor.shape[0] / batch_size]
        num_tokens = min(candidates) if candidates else self.graph_target_num_tokens[-1]

        current_num_all_tokens = flashinfer_metadata.num_all_tokens

        with torch.cuda.nvtx.range("ModelExecutor._run_model_graph.plan"):
            self.captured_models[batch_size][num_tokens].prefill_wrapper.plan(
                qo_indptr=flashinfer_metadata.input_ids_indptr,
                paged_kv_indptr=flashinfer_metadata.paged_kv_indptr,
                paged_kv_indices=flashinfer_metadata.paged_kv_indices,
                paged_kv_last_page_len=flashinfer_metadata.paged_kv_last_page_len,
                num_qo_heads=self.hf_config.num_attention_heads,
                num_kv_heads=self.hf_config.num_key_value_heads,
                head_dim_qk=self.hf_config.hidden_size // self.hf_config.num_attention_heads if not "qwen3" in self.hf_config.name_or_path.lower() else self.hf_config.head_dim,
                page_size=self.hf_config.kv_cache_block_size,
                causal=True,
                pos_encoding_mode="NONE", # TODO: Check
                use_fp16_qk_reduction=False,
                q_data_type=self.hf_config.torch_dtype,
                kv_data_type=self.past_key_values_dtype,
            )
        
        with torch.cuda.nvtx.range("ModelExecutor._run_model_graph.copy_input_tensors"):
            if input_ids_tensor.shape[0] == self.captured_models[batch_size][num_tokens].input_tensors[0].shape[0]:
                self.captured_models[batch_size][num_tokens].input_tensors[0].copy_(input_ids_tensor, non_blocking=True)
                self.captured_models[batch_size][num_tokens].input_tensors[1].copy_(position_ids_tensor, non_blocking=True)
                if self.eagle3_enabled and hidden_states is not None:
                    self.captured_models[batch_size][num_tokens].input_tensors[2].copy_(hidden_states, non_blocking=True)
            else:
                self.captured_models[batch_size][num_tokens].input_tensors[0][:current_num_all_tokens].copy_(input_ids_tensor, non_blocking=True)
                self.captured_models[batch_size][num_tokens].input_tensors[1][:current_num_all_tokens].copy_(position_ids_tensor, non_blocking=True)
                self.captured_models[batch_size][num_tokens].input_tensors[0][current_num_all_tokens:].zero_()
                self.captured_models[batch_size][num_tokens].input_tensors[1][current_num_all_tokens:].zero_()
                if self.eagle3_enabled and hidden_states is not None:
                    self.captured_models[batch_size][num_tokens].input_tensors[2][:current_num_all_tokens].copy_(hidden_states, non_blocking=True)
                    self.captured_models[batch_size][num_tokens].input_tensors[2][current_num_all_tokens:].zero_()
        
        with torch.cuda.nvtx.range("ModelExecutor._run_model_graph.copy_flashinfer_metadata"):
            self.captured_models[batch_size][num_tokens].flashinfer_metadata.copy_(
                flashinfer_metadata,
                prefill_wrapper=self.captured_models[batch_size][num_tokens].prefill_wrapper,
                kv_cache_block_size=self.hf_config.kv_cache_block_size
            )

        torch.cuda.synchronize()
        self.captured_models[batch_size][num_tokens].graph.replay()
        torch.cuda.synchronize()
        return self.captured_models[batch_size][num_tokens].output_tensor

    @profile_nvtx("ModelExecutor.run_sampler")
    def run_sampler(
            self, 
            logits: torch.Tensor, 
            input_ids_indptr: np.ndarray,
            sampling_params: List[SamplingParams] = None,
            return_log_probs: bool = False
        ) -> Tuple[List[np.ndarray], List[torch.Tensor]]:

        sampled_tokens, probs = self.sampler.forward(
            logits=logits,
            sampling_params=sampling_params[0] if sampling_params is not None else None,
            eos_token_ids=self.registry.get('val.model.eos_token_ids'),
            return_log_probs=return_log_probs
        )

        probs_per_request = probs
        sampled_tokens = sampled_tokens.cpu().numpy()
        sampled_tokens_per_request = np.split(sampled_tokens, input_ids_indptr[1:-1])

        # Check for negative values in sampled tokens
        for i, tokens in enumerate(sampled_tokens_per_request):
            if np.any(tokens < 0):
                print(f"Negative tokens found in request {i}: {tokens}")
                import pdb; pdb.set_trace()

        if DEBUG:
            print_debug(
                function_name="ModelIOProcessor.run_sampler",
                probs_shape=probs.shape,
                sampled_tokens_per_request_shape=[sampled_tokens.shape for sampled_tokens in sampled_tokens_per_request],
                probs_per_request_shape=[probs.shape for probs in probs_per_request]
            )
        return sampled_tokens_per_request, probs_per_request
    
    def eagle3_vocab_mapping(self, index):
        return index + self.model.get_vocab_mapping(index).to(index.device)

class TensorParallelModelExecutor(ModelExecutor):
    def __init__(
        self,
        registry: EngineRegistry,
        type=None,
        parallel_status: Optional[ParallelStatus] = None,
        local_rank: int = -1,
    ):
        print(f"TensorParallelModelExecutor INIT: {local_rank}")
        parallel_status.local_rank = local_rank
        self.parallel_status = parallel_status
        self.torch_process_group_port = str(registry.get('val.engine.torch_process_group_port'))[::-1]
        self.executor_parallel_group = self.init_dist_env(parallel_status)
        super().__init__(registry, type, parallel_status)
        self.local_max_num_blocks = self.registry.get('val.engine.num_kv_cache_blocks') // self.executor_parallel_group.world_size
        self.use_ray_executor = self.registry.get('val.engine.use_ray_executor')
        self.use_ray_worker = self.registry.get('val.engine.use_ray_worker')

    def get_node_and_gpu_ids(self) -> Tuple[str, List[int]]:
        node_id = ray.get_runtime_context().get_node_id()
        gpu_ids = ray.get_gpu_ids()
        return node_id, gpu_ids
    
    def get_node_ip(self) -> str:
        # return get_ip()
        return "127.0.0.1"

    def init_dist_env(self, parallel_status):
        assert not torch.distributed.is_initialized()
        assert torch.cuda.is_available()
        assert torch.cuda.device_count() > 0

        executor_parallel_group = ExecutorParallelGroup(
            global_world_size=len(parallel_status.device_ids),#parallel_status.global_world_size,
            global_ranks=list(range(len(parallel_status.device_ids))),# parallel_status.global_ranks,
            device_ids=parallel_status.device_ids,
            local_rank=parallel_status.local_rank,
            tensor_parallel_size=parallel_status.tensor_parallel_size,
            pipeline_parallel_size=parallel_status.pipeline_parallel_size,
        )

        print_debug(
            function_name="_init_dist_env",
            global_world_size=executor_parallel_group.global_world_size,
            global_ranks=executor_parallel_group.global_ranks,
            device_ids=executor_parallel_group.device_ids,
            local_rank=executor_parallel_group.local_rank,
            tensor_parallel_size=executor_parallel_group.tensor_parallel_size,
            pipeline_parallel_size=executor_parallel_group.pipeline_parallel_size,
        )

        # torch.distributed.destroy_process_group()
        torch.distributed.init_process_group(
            backend="nccl",
            init_method=f"tcp://{self.get_node_ip()}:{self.torch_process_group_port}",
            world_size=executor_parallel_group.global_world_size,
            rank=self.parallel_status.local_rank,
        )

        executor_parallel_group.build_model_parallel_groups()
        return executor_parallel_group
    
    def override_env_vars(self, vars: Dict[str, str]):
        os.environ.update(vars)

    def ensure_init_done(self):
        time.sleep(1)
        return True

    def load_model(self):
        # if self.registry.get('val.engine.hf_model_token') is not None:
        #     from  huggingface_hub import login
        #     login(self.registry.get('val.engine.hf_model_token'))
        
        self.non_graph_prefill_wrapper = self.build_prefill_wrapper(use_cuda_graph=False)
        
        self.non_graph_wrappers = {}
        for batch_size in self.graph_target_batch_sizes:
            self.non_graph_wrappers[batch_size] = self.build_prefill_wrapper(use_cuda_graph=False, batch_size=batch_size)
        
        for batch_size in self.graph_target_batch_sizes:
            for num_tokens in self.graph_target_num_tokens:
                self.captured_models[batch_size][num_tokens] = CUDAGraphCache(
                    batch_size=batch_size,
                    num_tokens=num_tokens,
                    prefill_wrapper=self.build_prefill_wrapper(use_cuda_graph=True, batch_size=batch_size)
                )

        if self.executor_parallel_group.world_size > 1:
            model, hf_config = load_or_create_tp_model(
                model_cls=self.model_cls,
                model_name_or_path=self.model_name_or_path,
                cache_dir=self.model_cache_dir,
                tensor_parallel_size=self.executor_parallel_group.tensor_parallel_size,
                execute_rank=0,  # Use rank 0 as execute rank
                config=self.hf_config,
                dtype=self.dtype,
                token=self.registry.get('val.engine.hf_model_token'),
                local_rank=self.executor_parallel_group.local_rank,
                world_size=self.executor_parallel_group.world_size,
            )
            for name, module in model.named_modules():
                if hasattr(module, "parallel_group"):
                    setattr(module, "parallel_group", self.executor_parallel_group)
            self.hf_config = hf_config
        
            setattr(self.hf_config, 'num_draft_steps', self.registry.get('val.engine.num_draft_steps'))
            setattr(self.hf_config, 'max_batch_size', self.registry.get('val.engine.num_max_batch_requests'))
            setattr(self.hf_config, 'kv_cache_block_size', self.kv_cache_block_size)
            setattr(self.hf_config, 'past_key_values_dtype', self.past_key_values_dtype)
            setattr(self.hf_config, 'torch_dtype', self.dtype)
            setattr(self.hf_config, 'past_key_values_dtype', self.past_key_values_dtype)

        else:
            try:
                model = self.model_cls.from_pretrained(self.model_name_or_path, torch_dtype=self.dtype, config=self.hf_config, cache_dir=self.model_cache_dir, use_safetensors=True, token=self.registry.get('val.engine.hf_model_token'))
            except Exception as e:
                model = self.model_cls.from_pretrained(self.model_name_or_path, torch_dtype=self.dtype, config=self.hf_config, cache_dir=self.model_cache_dir, use_safetensors=False, token=self.registry.get('val.engine.hf_model_token'))
        
        model.eval()
        model.to(self.device)
        
        # model.combine_qkv()

        # if self.quantization_type is None:
        #     model.combine_qkv()

        self.build_key_value_cache_blocks() # lazy init

        if DEBUG:
            print(f"Model {self.model_name_or_path} is loaded on device {self.device}")
        return model

    @profile_nvtx("TensorParallelModelExecutor.run_model")
    def run_model(
            self, 
            input_ids_tensor: torch.Tensor,
            position_ids_tensor: torch.Tensor,
            flashinfer_metadata: FlashInferMetadata,
            sampling_params: Optional[List[SamplingParams]] = None,
            num_logits_to_keep: Optional[int] = None,
            hidden_states: Optional[torch.Tensor] = None,
            **kwargs
        ):

        # print_debug(
        #     function_name="TensorParallelModelExecutor.run_model",
        #     probs_input_ids_indptr=flashinfer_metadata.input_ids_indptr_cpu,
        #     input_ids_lengths=flashinfer_metadata.input_ids_lengths.numpy(),
        #     input_ids_tensor_shape=input_ids_tensor.shape,
        #     flashinfer_metadata_input_ids_indptr_shape=flashinfer_metadata.input_ids_indptr.shape,
        # )

        if self.use_data_parallel_draft and self.model_type == 'model':
            torch.distributed.barrier()
            input_ids_tensor, position_ids_tensor, flashinfer_metadata, gpu_ids_indptr, gpu_ids_request_indptr = self.all_gather_model_inputs(
                input_ids_tensor,
                position_ids_tensor,
                flashinfer_metadata,
            )

            probs_input_ids_indptr  = flashinfer_metadata.input_ids_indptr_cpu
            input_ids_lengths = flashinfer_metadata.input_ids_lengths.numpy()
            # print_debug(
            #     function_name="TensorParallelModelExecutor.run_model",
            #     flashinfer_metadata_input_ids_indptr=flashinfer_metadata.input_ids_indptr,
            #     flashinfer_metadata_input_ids_lengths=flashinfer_metadata.input_ids_lengths,
            #     gpu_ids_indptr=gpu_ids_indptr,
            #     gpu_ids_request_indptr=gpu_ids_request_indptr,
            # )
        elif self.use_ray_worker:
            torch.distributed.barrier()
            input_ids_tensor, position_ids_tensor, flashinfer_metadata = self.broadcast_model_inputs(
                input_ids_tensor,
                position_ids_tensor,
                flashinfer_metadata
            )
            probs_input_ids_indptr = flashinfer_metadata.input_ids_indptr_cpu
            input_ids_lengths = flashinfer_metadata.input_ids_lengths.numpy()
        else:
            probs_input_ids_indptr = flashinfer_metadata.input_ids_indptr_cpu
            input_ids_lengths = flashinfer_metadata.input_ids_lengths.numpy()
        
        if self.eagle3_enabled and self.model_type == "draft_model" and hidden_states.shape[-1] != self.hf_config.hidden_size:
            hidden_states = self.model.process_hidden_states(hidden_states)
        
        output = super()._run_model(
            input_ids_tensor=input_ids_tensor,
            position_ids_tensor=position_ids_tensor,
            flashinfer_metadata=flashinfer_metadata,
            hidden_states=hidden_states,
            **kwargs
        )
        if self.eagle3_enabled and type(output) is tuple:
            hidden_states = output[0]
        else:
            hidden_states = output
        
        start_layer_idx = kwargs.get('start_layer_idx', None)
        end_layer_idx = kwargs.get('end_layer_idx', None)
        

        if num_logits_to_keep is not None:
            if num_logits_to_keep == 0:
                return [], [], None
            
            elif num_logits_to_keep == 1:
                slice_indices = probs_input_ids_indptr[1:] - 1
                cuda_indices = torch.from_numpy(slice_indices).to(hidden_states.device)

                hidden_states = hidden_states.index_select(0, cuda_indices)
                probs_input_ids_indptr = np.arange(probs_input_ids_indptr.shape[0])

            else:
                # LayerSkip
                if end_layer_idx is not None and  start_layer_idx < end_layer_idx:
                    end_indices = flashinfer_metadata.input_ids_indptr_cpu[1:]-1
                else:
                    end_indices = probs_input_ids_indptr[1:]

                num_tokens = np.minimum(input_ids_lengths, num_logits_to_keep)
                start_indices = end_indices - num_tokens
                slice_indices = np.concatenate([np.arange(s, e) for s, e in zip(start_indices, end_indices)])
                cuda_indices = torch.from_numpy(slice_indices).to(hidden_states.device)
                hidden_states = hidden_states.index_select(0, cuda_indices)
                probs_input_ids_indptr = np.concatenate([[0], np.cumsum(num_tokens)])
                # print_debug(
                #     function_name="TensorParallelModelExecutor.run_model.num_logits_to_keep",
                #     num_tokens=num_tokens,
                #     start_indices=start_indices,
                #     end_indices=end_indices,
                #     slice_indices=slice_indices,
                #     cuda_indices=cuda_indices,
                #     hidden_states_shape=hidden_states.shape,
                #     probs_input_ids_indptr=probs_input_ids_indptr,
                # )

        logits = self.get_logits(hidden_states)
        # print(f"self.local_rank: {self.executor_parallel_group.local_rank}")

        if self.use_data_parallel_draft and self.model_type == 'model':
            # start_idx = gpu_ids_indptr[self.local_rank]
            # end_idx = gpu_ids_indptr[self.local_rank + 1]

            start_request_idx = gpu_ids_request_indptr[self.local_rank]
            end_request_idx = gpu_ids_request_indptr[self.local_rank + 1]
            start_idx = probs_input_ids_indptr[start_request_idx]
            end_idx = probs_input_ids_indptr[end_request_idx]
                
            if start_idx == end_idx:
                return [], [], None
            
            # cuda_indices = torch.from_numpy(np.arange(start_idx, end_idx)).to(logits.device)

            logits = logits[start_idx:end_idx, :]
            probs_input_ids_indptr = probs_input_ids_indptr[start_request_idx:end_request_idx+1] - probs_input_ids_indptr[start_request_idx]

            # print_debug(
            #     function_name="TensorParallelModelExecutor.run_model.last",
            #     start_request_idx=start_request_idx,
            #     end_request_idx=end_request_idx,
            #     probs_input_ids_indptr = probs_input_ids_indptr,
            #     start_idx=start_idx,
            #     end_idx=end_idx,
            #     hidden_states_shape=logits.shape,
            # )


        elif self.executor_parallel_group.local_rank != 0:
            return [], [], None

        sampled_tokens_per_request, probs = self.run_sampler(
            logits,
            input_ids_indptr=probs_input_ids_indptr, # local_request_indices,
            sampling_params=sampling_params,
            return_log_probs=False
        )
        del logits

        # if self.model_type == "draft_model":
        #     print(f"\n===================={self.model_type} run_sampler: ====================")
        #     print(f"model: {self.model_type} | ray worker {self.executor_parallel_group.global_rank} | device {self.executor_parallel_group.device_id}")
        #     print(f"torch.cuda.memory(current): {torch.cuda.memory_stats(self.device)['allocated_bytes.all.current']/1024/1024} MB")
        #     print(f"torch.cuda.memory(reserved current): {torch.cuda.memory_stats(self.device)['reserved_bytes.all.current']/1024/1024} MB")
        #     print(f"==========================================================================")
        if self.use_ray_executor:
            probs = get_tensor_ipc_info(probs)
        # probs_per_request = [get_tensor_ipc_info(probs) for probs in probs_per_request]
        
        if self.eagle3_enabled:
            if type(output) is tuple:
                hidden_states = torch.cat(output[1], dim=-1)
            hidden_states_list = torch.split(hidden_states, input_ids_lengths.tolist(), dim=0)
            return sampled_tokens_per_request, probs, probs_input_ids_indptr, hidden_states_list
        return sampled_tokens_per_request, probs, probs_input_ids_indptr

    
    def broadcast_model_inputs(self, input_ids_tensor, position_ids_tensor, flashinfer_metadata):
       
        # CPU Broadcast
        if self.local_rank == 0:
            input_ids = input_ids_tensor.numpy()
            position_ids = position_ids_tensor.numpy()
            flashinfer_metadata.to_cpu()
            objects = [input_ids, position_ids, flashinfer_metadata]
        else:
            objects = [None, None, None]

        torch.distributed.broadcast_object_list(objects, src=0)
        input_ids_tensor = torch.from_numpy(objects[0])
        position_ids_tensor = torch.from_numpy(objects[1])
        flashinfer_metadata = objects[2]
        return input_ids_tensor, position_ids_tensor, flashinfer_metadata

        # GPU Broadcast
        # if self.local_rank == 0:
        #     flashinfer_metadata.to_cpu()
        #     input_ids_tensor = input_ids_tensor.to(self.device, non_blocking=True)
        #     position_ids_tensor = position_ids_tensor.to(self.device, non_blocking=True)
        #     shapes = [input_ids_tensor.shape, position_ids_tensor.shape]
        # else:
        #     shapes = [None, None]
                  
        # torch.distributed.broadcast_object_list(shapes, src=0)
        # input_ids_tensor_shape, position_ids_tensor_shape = shapes

        # if self.local_rank != 0:
        #     input_ids_tensor = torch.empty(input_ids_tensor_shape, dtype=torch.int32, device=self.device)
        #     position_ids_tensor = torch.empty(position_ids_tensor_shape, dtype=torch.int32, device=self.device)
        
        # broadcast_results = [input_ids_tensor, position_ids_tensor]
        # torch.cuda.synchronize()
        
        # handle_input_ids = torch.distributed.broadcast(input_ids_tensor, src=0, async_op=True)
        # handle_position_ids = torch.distributed.broadcast(position_ids_tensor, src=0, async_op=True)
        # if self.local_rank == 0:
        #     objects = [flashinfer_metadata]
        # else:
        #     objects = [None]
        # torch.distributed.broadcast_object_list(objects, src=0)
        
        # handle_input_ids.wait()
        # handle_position_ids.wait()
        # flashinfer_metadata = objects[0]

        # return input_ids_tensor, position_ids_tensor, flashinfer_metadata


    def all_gather_model_inputs(
            self,
            input_ids_tensor, 
            position_ids_tensor, 
            flashinfer_metadata
        ):
        # allgather requests
        # objs = [input_ids_tensor, position_ids_tensor, flashinfer_metadata] or [None, None, None]
        # The index of 'allgather_results' must be the rank of each process
        #print_debug(
        #    function_name="TensorParallelModelExecutor.all_gather_model_inputs",
        #    self_local_max_num_blocks=self.local_max_num_blocks,
        #    self_executor_parallel_group_world_size=self.executor_parallel_group.world_size,
        #    flashinfer_metadata_input_ids_indptr_device=flashinfer_metadata.input_ids_indptr.device
        #)
        allgather_results = [None for _ in range(self.executor_parallel_group.world_size)]
        # flashinfer_metadata.to_numpy()
        
        # if is_prefill:
        flashinfer_metadata.to_cpu()

        torch.distributed.all_gather_object(allgather_results, [input_ids_tensor, position_ids_tensor, flashinfer_metadata])
        # torch.distributed.all_gather_object_list(allgather_results, [input_ids_tensor.to_numpy(), position_ids_tensor.to_numpy(), flashinfer_metadata])


        # Concatenate input_ids_tensor and position_ids_tensor
        input_ids_tensor_list = []
        position_ids_tensor_list = []
        for rank in range(len(allgather_results)):
            if allgather_results[rank] == None:
                continue
    
            input_ids_tensor_list.append(allgather_results[rank][0])
            position_ids_tensor_list.append(allgather_results[rank][1])
    
        input_ids_tensor = torch.cat(input_ids_tensor_list, dim=0)
        position_ids_tensor = torch.cat(position_ids_tensor_list, dim=0)
    
        # Concatenate flashinfer metadata
        input_ids_indptr_list = []
        input_ids_indptr_list.append(torch.zeros((1, ), dtype=torch.int32))
        input_ids_lengths_list = []
        paged_kv_indices_list = []
        paged_kv_indptr_list = []
        paged_kv_indptr_list.append(torch.zeros((1, ), dtype=torch.int32))
        paged_kv_last_page_len_list = []
        last_input_ids_indptr = 0
        last_paged_kv_indptr = 0
        gpu_ids_indptr = np.empty((self.executor_parallel_group.world_size + 1, ))
        gpu_ids_indptr[0] = 0

        last_requests_indptr = 0
        gpu_ids_request_indptr = np.empty((self.executor_parallel_group.world_size + 1, ))
        gpu_ids_request_indptr[0] = 0

        for rank in range(len(allgather_results)):
            if allgather_results[rank] == None:
                continue
    
            flashinfer_metadata = allgather_results[rank][2]
    
            input_ids_lengths_list.append(flashinfer_metadata.input_ids_lengths)
            paged_kv_indices_list.append(flashinfer_metadata.paged_kv_indices+(rank*self.local_max_num_blocks))
            paged_kv_last_page_len_list.append(flashinfer_metadata.paged_kv_last_page_len)
    
            if len(flashinfer_metadata.input_ids_indptr) != 0:
                input_ids_indptr_list.append(flashinfer_metadata.input_ids_indptr[1:]+last_input_ids_indptr)
                last_input_ids_indptr += flashinfer_metadata.input_ids_indptr[-1].item()
                last_requests_indptr += flashinfer_metadata.input_ids_indptr.shape[0]-1
            gpu_ids_indptr[rank + 1] = last_input_ids_indptr
            gpu_ids_request_indptr[rank + 1] = last_requests_indptr
    
            
            if len(flashinfer_metadata.paged_kv_indptr) != 0:
                paged_kv_indptr_list.append(flashinfer_metadata.paged_kv_indptr[1:]+last_paged_kv_indptr)
                last_paged_kv_indptr += flashinfer_metadata.paged_kv_indptr[-1].item()
    
        # TODO: Please import FlashInferMetadata and kv_cache_block_size
        flashinfer_metadata = FlashInferMetadata(
            input_ids_indptr=torch.cat(input_ids_indptr_list, dim=0),
            input_ids_lengths=torch.cat(input_ids_lengths_list, dim=0),
            batch_indices=None,
            positions=None,
            paged_kv_indices=torch.cat(paged_kv_indices_list, dim=0),
            paged_kv_indptr=torch.cat(paged_kv_indptr_list, dim=0),
            paged_kv_last_page_len=torch.cat(paged_kv_last_page_len_list, dim=0),
            kv_cache_block_size=self.kv_cache_block_size
        )

        #print_debug(
        #    function_name="TensorParallelModelExecutor.all_gather_model_inputs",
        #    gpu_ids_indptr=gpu_ids_indptr.astype(np.int32),
        #    input_ids_tensor=input_ids_tensor,
        #    position_ids_tensor=position_ids_tensor,
        #    input_ids_indptr=torch.cat(input_ids_indptr_list, dim=0),
        #    input_ids_lengths=torch.cat(input_ids_lengths_list, dim=0),
        #    paged_kv_indices=torch.cat(paged_kv_indices_list, dim=0),
        #    paged_kv_indptr=torch.cat(paged_kv_indptr_list, dim=0),
        #    paged_kv_last_page_len=torch.cat(paged_kv_last_page_len_list, dim=0),
        #)

        gpu_ids_indptr = gpu_ids_indptr.astype(np.int32)
        gpu_ids_request_indptr = gpu_ids_request_indptr.astype(np.int32)
    
        return input_ids_tensor, position_ids_tensor, flashinfer_metadata, gpu_ids_indptr, gpu_ids_request_indptr















def get_partition_indices(input_ids_indptr, local_rank, world_size):
    total_tokens = input_ids_indptr[-1]
    target_tokens = total_tokens / world_size
    
    rank_start = int(target_tokens * local_rank)
    rank_end = int(target_tokens * (local_rank + 1)) if local_rank < world_size - 1 else total_tokens
    
    # 각 rank가 처리할 request 범위 찾기
    start_request_idx = np.searchsorted(input_ids_indptr, rank_start, side='right') - 1
    start_idx = input_ids_indptr[start_request_idx]
    
    end_request_idx = np.searchsorted(input_ids_indptr, rank_end, side='right') - 1
    end_idx = input_ids_indptr[end_request_idx + 1]
    
    return start_idx, end_idx

def partition_logits(logits, input_ids_indptr, input_ids_lengths, local_rank, world_size):
    """
    GPU 텐서를 효율적으로 분할합니다.
    
    Args:
        logits: torch.Tensor, shape [num_tokens, hidden_size]
        input_ids_indptr: np.ndarray
        input_ids_lengths: np.ndarray
        local_rank: int
        world_size: int
    
    Returns:
        local_logits: torch.Tensor, 현재 rank의 logits
        local_request_indices: List[int], local_logits 내 각 request의 시작 인덱스
    """
    start_idx, end_idx = get_partition_indices(
        input_ids_indptr, local_rank, world_size
    )
    
    # GPU 텐서 슬라이싱
    logits = logits[start_idx:end_idx]
    
    # local logits 내에서의 request 경계 계산 (시작점 0 포함, 끝점 포함)
    local_request_indices = [0] + [
        idx - start_idx for idx in input_ids_indptr 
        if idx > start_idx and idx <= end_idx
    ]

    if local_request_indices[-1] == 0:
        return None, None
    
    return logits, local_request_indices


def prepare_data_parallel_inputs(
        input_ids_tensor,
        position_ids_tensor,
        flashinfer_metadata,
        num_logits_to_keep,
        **kwargs
):
    pass
