from typing import List, Dict, Optional, Tuple
import os
import time

import torch
import torch.distributed as dist

from sdprofiler.utils.parallel_group_utils import ExecutorParallelGroup
from sdprofiler.models.layers.linear import TensorParallelEmbedding, TensorParallelLearnedPositionalEmbedding, RowParallelLinear, ColumnParallelLinear


def _get_last_part(name: str) -> str:
    return name.split(".")[-1]


def _get_second_last_part(name: str) -> Optional[str]:
    parts = name.split(".")
    return parts[-2] if len(parts) > 1 else None


def _contains_pattern(name: str, patterns: List[str]) -> bool:
    return any(pattern in _get_last_part(name) for pattern in patterns)


def _is_position_embedding(name: str) -> bool:
    return _get_last_part(name) == "embed_positions"


def _is_self_attention(name: str) -> bool:
    return _get_second_last_part(name) == "self_attn" or "attention" in name


def _is_lm_head(name: str) -> bool:
    return _get_last_part(name) == "lm_head" or "embed_out" in name


def _is_attention_projection(name: str) -> bool:
    return _contains_pattern(name, ["q_proj", "k_proj", "v_proj", "query_key_value"])


def _is_mlp_up_projection(name: str) -> bool:
    return _contains_pattern(name, ["up_proj", "gate_proj"])


def _is_embedding(module: torch.nn.Module) -> bool:
    return isinstance(module, (torch.nn.Embedding, torch.nn.EmbeddingBag))


def _is_linear(module: torch.nn.Module) -> bool:
    return isinstance(module, torch.nn.Linear)


def convert_model_config(model: torch.nn.Module, tp_degree: int):
    adjust_config_name = ["embed_dim", "hidden_size", "num_heads", "num_key_value_heads"]
    for name, module in model.named_modules():
        for config_name in adjust_config_name:
            if hasattr(module, config_name):
                config_value = getattr(module, config_name)
                setattr(module, config_name, config_value // tp_degree)
    
    adjust_config_name = ["n_embed", "hidden_size", "num_attention_heads", "num_key_value_heads"]
    for config_name in adjust_config_name:
        if hasattr(model.config, config_name):
            config_value = getattr(model.config, config_name)
            setattr(model.config, config_name, config_value // tp_degree)

    return model


def get_split_weights(model: torch.nn.Module, world_size: int) -> Dict:
    split_weights = {}
    for name, module in model.named_modules():
        if _is_embedding(module):
            assert module.max_norm is None or module.norm_type < 2
            if not _is_position_embedding(name):
                if hasattr(module, 'weight') and module.weight is not None:
                    split_weights[name] = {}
                    split_weights[name]['weight'] = torch.tensor_split(module.weight, world_size, dim=1)
        elif _is_linear(module):
            if hasattr(module, 'weight') and module.weight is not None:
                split_weights[name] = {}
                if _is_self_attention(name):
                    if _is_attention_projection(name):
                        split_weights[name]['weight'] = torch.tensor_split(module.weight, world_size, dim=0)
                        if hasattr(module, "bias") and module.bias is not None:
                            split_weights[name]['bias'] = torch.tensor_split(module.bias, world_size, dim=0)
                    else:  # This is output projection layer
                        split_weights[name]['weight'] = torch.tensor_split(module.weight, world_size, dim=1)
                elif _is_lm_head(name):
                    split_weights[name]['weight'] = torch.tensor_split(module.weight, world_size, dim=0)
                else:
                    if _is_mlp_up_projection(name):
                        split_weights[name]['weight'] = torch.tensor_split(module.weight, world_size, dim=0)
                        if hasattr(module, "bias") and module.bias is not None:
                            split_weights[name]['bias'] = torch.tensor_split(module.bias, world_size, dim=0)
                    else:  # This is down projection layer
                        split_weights[name]['weight'] = torch.tensor_split(module.weight, world_size, dim=1)
    return split_weights


def convert_to_tensor_parallel_with_weights(model: torch.nn.Module, weights: Dict, rank: int = 0):
    modules_to_replace = {}
    for name, module in model.named_modules():
        if name in weights:
            if _is_embedding(module):
                if not _is_position_embedding(name):
                    split_weight_tensor = weights[name]['weight'][rank]
                    modules_to_replace[name] = TensorParallelEmbedding(
                        split_weight_tensor, 
                        module.padding_idx, 
                        None  # Will be set later
                    )
            elif _is_linear(module):
                if _is_self_attention(name):
                    if _is_attention_projection(name):
                        split_weight_tensor = weights[name]['weight'][rank]
                        split_bias_tensor = None
                        if hasattr(module, "bias") and module.bias is not None:
                            split_bias_tensor = weights[name]['bias'][rank]
                        modules_to_replace[name] = ColumnParallelLinear(
                            split_weight_tensor, 
                            split_bias_tensor, 
                            None  # Will be set later
                        )
                    else:  # This is output projection layer
                        split_weight_tensor = weights[name]['weight'][rank]
                        modules_to_replace[name] = RowParallelLinear(
                            split_weight_tensor, 
                            module.bias, 
                            None  # Will be set later
                        )
                elif _is_lm_head(name):
                    split_weight_tensor = weights[name]['weight'][rank]
                    modules_to_replace[name] = ColumnParallelLinear(
                        split_weight_tensor, 
                        module.bias, 
                        None,  # Will be set later
                        gather_output=True
                    )
                else:
                    if _is_mlp_up_projection(name):
                        split_weight_tensor = weights[name]['weight'][rank]
                        split_bias_tensor = None
                        if hasattr(module, "bias") and module.bias is not None:
                            split_bias_tensor = weights[name]['bias'][rank]
                        modules_to_replace[name] = ColumnParallelLinear(
                            split_weight_tensor, 
                            split_bias_tensor, 
                            None  # Will be set later
                        )
                    else:  # This is down projection layer
                        split_weight_tensor = weights[name]['weight'][rank]
                        modules_to_replace[name] = RowParallelLinear(
                            split_weight_tensor, 
                            module.bias, 
                            None  # Will be set later
                        )
    
    # Replace the modules after iteration
    for name, new_module in modules_to_replace.items():
        module_path = name.split('.')
        parent_module = model
        for part in module_path[:-1]:
            parent_module = getattr(parent_module, part)
        setattr(parent_module, module_path[-1], new_module)
    
    return model


def get_tp_cache_path(model_name_or_path: str, cache_dir: str, tensor_parallel_size: int, rank: int) -> str:
    model_name = os.path.basename(model_name_or_path.rstrip('/'))
    tp_cache_dir = os.path.join(cache_dir, "tp_shards")
    return os.path.join(tp_cache_dir, f"{model_name}_tp_{tensor_parallel_size}", f"rank_{rank}")


def get_tp_completion_file(model_name_or_path: str, cache_dir: str, tensor_parallel_size: int) -> str:
    model_name = os.path.basename(model_name_or_path.rstrip('/'))
    tp_cache_dir = os.path.join(cache_dir, "tp_shards")
    return os.path.join(tp_cache_dir, f"{model_name}_tp_{tensor_parallel_size}", "tp_creation_complete")


def save_tp_shard_to_cache(model, cache_path: str, rank: int):
    os.makedirs(cache_path, exist_ok=True)
    torch.save(model, os.path.join(cache_path, "model.pt"))
    print(f"[Rank {rank}]: Saved TP shard to {cache_path}")


def load_tp_shard_from_cache(cache_path: str):
    model = torch.load(os.path.join(cache_path, "model.pt"), weights_only=False, map_location='cpu')
    return model, model.config


def create_and_save_tp_shards(model, model_name_or_path: str, cache_dir: str, 
                             tensor_parallel_size: int, execute_rank: int, 
                             local_rank: int):
    completion_file = get_tp_completion_file(model_name_or_path, cache_dir, tensor_parallel_size)
    
    print(f"Creating TP shards by rank {local_rank}")
    
    try:
        split_weights = get_split_weights(model, tensor_parallel_size)
        model = convert_model_config(model, tensor_parallel_size)
        
        for rank in range(execute_rank, execute_rank + tensor_parallel_size):
            rank_model = convert_to_tensor_parallel_with_weights(model, split_weights, rank - execute_rank)
            
            cache_path = get_tp_cache_path(model_name_or_path, cache_dir, tensor_parallel_size, rank)
            save_tp_shard_to_cache(rank_model, cache_path, rank)
        
        with open(completion_file, 'w') as f:
            f.write('complete')
        print(f"TP shards created successfully")
        
        current_cache_path = get_tp_cache_path(model_name_or_path, cache_dir, tensor_parallel_size, local_rank)
        return load_tp_shard_from_cache(current_cache_path)
        
    except Exception as e:
        print(f"Error creating TP shards: {e}")
        raise


@torch.no_grad()
def load_or_create_tp_model(model_cls, model_name_or_path: str, cache_dir: str, tensor_parallel_size: int, 
                           execute_rank: int, config, dtype, token, local_rank: int, world_size: int):
    """Load or create tensor parallel model with caching."""
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=local_rank, device_id=torch.device(f'cuda:{local_rank}'))
        print(f"Initialized distributed training on rank {local_rank} with world size {world_size}")
    
    tp_cache_path = get_tp_cache_path(model_name_or_path, cache_dir, tensor_parallel_size, local_rank)
    completion_file = get_tp_completion_file(model_name_or_path, cache_dir, tensor_parallel_size)
    
    if os.path.exists(tp_cache_path) and os.path.exists(completion_file):
        print(f"[Rank {local_rank}]: Loading TP shard from cache: {tp_cache_path}")
        return load_tp_shard_from_cache(tp_cache_path)
        
    if local_rank == execute_rank:
        print(f"[Rank {local_rank}]: Creating TP shards for tensor_parallel_size={tensor_parallel_size}")
        try:
            model = model_cls.from_pretrained(
                model_name_or_path,
                torch_dtype=dtype,
                config=config,
                cache_dir=cache_dir,
                use_safetensors=True,
                token=token
            )
        except OSError:
            model = model_cls.from_pretrained(
                model_name_or_path,
                cache_dir=cache_dir,
                torch_dtype=dtype,
                config=config,
                token=token
            )
        adjust_config_names = ["n_embed", "hidden_size", "num_attention_heads", "num_key_value_heads"]
        for config_name in adjust_config_names:
            if hasattr(config, config_name):
                config_value = getattr(config, config_name)
                if config_value % tensor_parallel_size != 0 or config_value < tensor_parallel_size:
                    print(f"[Rank {local_rank}]: {config_name} {config_value} is not divisible by tensor_parallel_size {tensor_parallel_size}")
                    exit()
        model, config = create_and_save_tp_shards(model, model_name_or_path, cache_dir, 
                                                 tensor_parallel_size, execute_rank, 
                                                 local_rank)
        
        print(f"[Rank {local_rank}]: Loaded TP shard from cache: {tp_cache_path}")
        if model is not None:
            return model, config
    
    print(f"[Rank {local_rank}]: Waiting for execute_rank to create TP shards...")
    while not os.path.exists(completion_file):
        time.sleep(0.1)
    
    model, config = load_tp_shard_from_cache(tp_cache_path)
    print(f"[Rank {local_rank}]: Loaded TP shard from cache: {tp_cache_path}")
    
    return model, config

