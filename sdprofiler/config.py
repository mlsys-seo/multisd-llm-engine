import os
import random

from dataclasses import dataclass, field
from typing import Optional, Any, List, Type
import torch

from transformers import AutoConfig, PreTrainedModel

@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    do_sample: bool = False
    min_output_length: int = 64
    max_output_length: int = 512
    repetition_penalty: float = None
    frequency_penalty: float = None
    presence_penalty: float = None
    return_hidden_states: bool = False


@dataclass
class EngineConfig:
    num_model_replicas: int = 1
    kv_cache_block_size: int = 16
    num_kv_cache_blocks: int = 4096
    num_max_batch_tokens: int = 2048
    num_max_batch_requests: int = 64
    num_min_batch_requests: int = None
    num_max_batch_requests_pipe: int = 64
    num_min_batch_requests_pipe: int = None

    max_context_len: int = None
    
    worker_class: type = None
    request_class: type = None
    
    model_cache_dir: str = None
    hf_model_token: str = None
    
    torch_process_group_port: int = 54321
    use_ray_executor: bool = False
    use_ray_worker: bool = False
    enable_nsight: bool = False
    use_chunked_prefill: bool = False
    use_cuda_graph: bool = False
    cuda_graph_target_batch_sizes: List[int] = None
    static_batch_profile: bool = False
    use_sparse_probs: bool = False
    
    num_draft_steps: int = 1 # only used for speculative engine
    num_verify_steps: int = 1 # only used for hierarchical speculative engine
    profile_prompt_len: int = 256

    draft_last_layer_idx: int = 0
    verify_last_layer_idx: int = 0

    no_cutoff: bool = False
    
    beta_threshold: float = 0.9
    run_profile_acceptance_prob: bool = False
    acceptance_profile_path: str = None
    use_only_beta_cutoff: bool = False
    use_data_parallel_draft: bool = False
    quantization: str = None

    # SmartSpec related configurations
    smart_spec_enabled: bool = False
    moving_average_window: int = 20

    # SVIP related configurations
    svip_enabled: bool = False
    threshold_for_svip: float = 0.5


    use_cache_offloading: bool = False
    
    # Eagle3 related configurations
    eagle3_enabled: bool = False
    
    def __post_init__(self):

        if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
            first_device_id = 0
        else:
            first_device_id = int(os.environ.get("CUDA_VISIBLE_DEVICES").split(",")[0])

        self.torch_process_group_port = 1234 + first_device_id

        if self.worker_class is None:
            from sdprofiler.worker import Worker # lazy import
            self.worker_class = Worker

        if self.request_class is None:
            from sdprofiler.request import Request # lazy import
            self.request_class = Request

        self.num_max_batch_requests_pipe = self.num_max_batch_requests

        # if self.num_min_batch_requests is None:
        #     if self.cuda_graph_target_batch_sizes is not None:
        #         self.num_min_batch_requests = self.cuda_graph_target_batch_sizes[-1]
        #     else:
        #         self.num_min_batch_requests = self.num_max_batch_requests

        if self.num_kv_cache_blocks is not None and self.max_context_len is not None:
            self.num_kv_cache_blocks = max(self.num_kv_cache_blocks, ((self.max_context_len + self.kv_cache_block_size) // self.kv_cache_block_size) * self.num_max_batch_requests + 16)

        if self.max_context_len is None:
            self.max_context_len = (self.num_kv_cache_blocks * self.kv_cache_block_size) // self.num_max_batch_requests

        if self.use_ray_worker:
            if self.use_data_parallel_draft:
                self.use_ray_executor = True
            else:
                self.use_ray_executor = False

        
        self.validate()

    def validate(self):
        assert self.num_model_replicas > 0, "num_model_replicas must be greater than 0"
        assert self.kv_cache_block_size > 0, "kv_cache_block_size must be greater than 0"
        assert self.num_kv_cache_blocks > 0, "num_kv_cache_blocks must be greater than 0"

@dataclass
class ModelParallelConfig:

    def __post_init__(self):
        self.world_size = self.pipeline_parallel_size * self.tensor_parallel_size

        if self.device_ids is None:
            self.device_ids = []
            for i in range(torch.cuda.device_count()):
                self.device_ids.append(torch.cuda._get_device_index(i))
            # self.device_ids = [i for i in range(torch.cuda.device_count())]

        self.device_ids = self.device_ids[:self.world_size]

        self.validate()

    def validate(self):
        assert self.world_size == len(self.device_ids), "world_size must be equal to the length of device_ids"
        assert self.pipeline_parallel_size > 0, "pipeline_parallel_size must be greater than 0"
        assert self.tensor_parallel_size > 0, "tensor_parallel_size must be greater than 0"
        assert len(self.device_ids) > 0, "device_ids must be set"

@dataclass
class ModelConfig:
    model_name_or_path: str
    type: Optional[str] = None  # Allowed values: None, 'draft', 'verify'
    cache_dir: str = None
    device: str = 'cuda'
    dtype: torch.dtype = torch.bfloat16

    hf_config: Optional[Any] = None
    hf_model_token: str = None
    max_context_len: int = None
    eos_token_ids: List[int] = field(default_factory=lambda: [])
    vocab_size: int = None
    model_class: Type[PreTrainedModel] = None


    data_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    tensor_parallel_size: int = 1
    world_size: int = None
    device_ids: List[int] = None
    
    eagle3_enabled: bool = False # does this work?

    def __post_init__(self):
        if self.hf_config is None:
            print(f"Loading config for {self.model_name_or_path}, token: {self.hf_model_token}")
            self.hf_config = AutoConfig.from_pretrained(self.model_name_or_path, token=self.hf_model_token)

        if self.max_context_len is None or 'opt' in self.model_name_or_path.lower():
            self.max_context_len = self.hf_config.max_position_embeddings
        else:
            self.hf_config.max_position_embeddings = self.max_context_len + 32

        if self.eos_token_ids is None or len(self.eos_token_ids) == 0:
            self.eos_token_ids = [self.hf_config.eos_token_id]
            if self.hf_config.pad_token_id is not None:
                self.eos_token_ids.append(self.hf_config.pad_token_id)

        if self.vocab_size is None:
            self.vocab_size = self.hf_config.vocab_size
        
        if self.model_class is None:
            if "qwen2" in self.model_name_or_path.lower() or 'qwen-2' in self.model_name_or_path.lower() or "qwen2" in self.hf_config.architectures[0].lower():
                from sdprofiler.models.qwen2 import Qwen2ForCausalLM
                self.model_class = Qwen2ForCausalLM
            elif "qwen3" in self.model_name_or_path.lower():
                if self.eagle3_enabled and not self.type == "verify":
                    # jkim: this is a temporary fix for the eagle3 draft model
                    if "AngelSlim" in self.model_name_or_path:
                        from sdprofiler.models.draft_eagle3 import Eagle3DraftLlamaForCausalLM
                        self.model_class = Eagle3DraftLlamaForCausalLM
                    else:
                        from sdprofiler.models.qwen3_eagle3 import Eagle3Qwen3ForCausalLM
                        self.model_class = Eagle3Qwen3ForCausalLM
                else:
                    from sdprofiler.models.qwen3 import Qwen3ForCausalLM
                    self.model_class = Qwen3ForCausalLM
            elif "layerskip" in self.model_name_or_path.lower():
                from sdprofiler.models.llama import LlamaForCausualLMWithSelfSpeculation
                self.model_class = LlamaForCausualLMWithSelfSpeculation
            elif "llama" in self.model_name_or_path.lower():
                if self.eagle3_enabled:
                    if "yuhuili" in self.model_name_or_path.lower():
                        from sdprofiler.models.draft_eagle3 import Eagle3DraftLlamaForCausalLM
                        self.model_class = Eagle3DraftLlamaForCausalLM
                    else:
                        from sdprofiler.models.llama_eagle3 import Eagle3LlamaForCausalLM
                        self.model_class = Eagle3LlamaForCausalLM
                else:
                    from sdprofiler.models.llama import LlamaForCausalLM
                    self.model_class = LlamaForCausalLM
            elif "opt" in self.model_name_or_path.lower():
                from sdprofiler.models.opt import OPTForCausalLM
                self.model_class = OPTForCausalLM
                setattr(self.hf_config, 'num_key_value_heads', self.hf_config.num_attention_heads)
            elif "pythia" in self.model_name_or_path.lower():
                from sdprofiler.models.gpt_neox import GPTNeoXForCausalLM
                self.model_class = GPTNeoXForCausalLM
                setattr(self.hf_config, 'num_key_value_heads', self.hf_config.num_attention_heads)
            else:
                raise ValueError(f"Model class for {self.model_name_or_path} is not implemented")
        
        if self.world_size is None:
            self.world_size = self.pipeline_parallel_size * self.tensor_parallel_size * self.data_parallel_size

        if self.device_ids is None:
            if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
                self.device_ids = [i for i in range(torch.cuda.device_count())]
            else:
                self.device_ids = [int(i) for i in os.environ.get("CUDA_VISIBLE_DEVICES").split(",")]

        self.device_ids = self.device_ids[:self.world_size]


        self.validate()

    def set_kv_cache_block_size(self, kv_cache_block_size: int):
        setattr(self.hf_config, 'kv_cache_block_size', kv_cache_block_size)

    def validate(self):
        assert self.model_name_or_path is not None, "model_name_or_path must be set"
    
        

@dataclass
class ParallelConfig:
    world_size: int = 1
    pipeline_parallel_size: int = 1
    tensor_parallel_size: int = 1

    def __post_init__(self):
        self.validate()

    def validate(self):
        assert self.world_size > 0, "world_size must be greater than 0"
        assert self.pipeline_parallel_size > 0, "pipeline_parallel_size must be greater than 0"
        assert self.tensor_parallel_size > 0, "tensor_parallel_size must be greater than 0"
