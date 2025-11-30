import os
import argparse
import shutil
import functools

import torch

try:
    import torch.cuda.nvtx as nvtx
    NVTX_AVAILABLE = True
except ImportError:
    NVTX_AVAILABLE = False

NVTX_ENV = os.environ.get("SDPROFILER_PROFILE_NVTX", "0")

try:
    # Use shutil.get_terminal_size with a fallback to prevent OSError
    SCREEN_WIDTH = shutil.get_terminal_size(fallback=(80, 24)).columns
except Exception as e:
    # Fallback in case shutil.get_terminal_size also fails
    SCREEN_WIDTH = 80
    print(f"Warning: Unable to get terminal size. Using default width {SCREEN_WIDTH}. Error: {e}")

def add_cli_args(parser: argparse.ArgumentParser = None) -> argparse.ArgumentParser:
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument('--model_cache_dir', type=str, default="/workspace/cache", help='Model cache directory.')
    parser.add_argument('--model_name_or_path', type=str, default="Qwen/Qwen2.5-7B-Instruct", help='Model name or path.')
    parser.add_argument('--tensor_parallel_size', type=int, default=1, help='Number of tensor parallel size.')
    parser.add_argument('--device_ids', nargs='+', type=int, default=None, help='Number of device ids.')
    parser.add_argument('--draft_model_name_or_path', type=str, default=None, help='Model name or path.')
    parser.add_argument('--draft_tensor_parallel_size', type=int, default=1, help='Number of draft tensor parallel size.')
    parser.add_argument('--draft_device_ids', nargs='+', type=int, default=None, help='Number of draft device ids.')
    parser.add_argument('--verify_model_name_or_path', type=str, default=None, help='Model name or path.')
    parser.add_argument('--verify_tensor_parallel_size', type=int, default=1, help='Number of verify tensor parallel size.')
    parser.add_argument('--verify_device_ids', nargs='+', type=int, default=None, help='Number of verify device ids.')
    parser.add_argument('--hf_model_token', type=str, default=None, help='Huggingface model token.')
    parser.add_argument('--num_draft_steps', type=int, default=None, help='Number of draft steps.')
    parser.add_argument('--num_verify_steps', type=int, default=None, help='Number of verify steps.')
    parser.add_argument('--max_new_tokens', type=int, default=None, help='Number of max new tokens.')
    parser.add_argument('--min_new_tokens', type=int, default=64, help='Number of min new tokens.')
    parser.add_argument('--do_sample', action='store_true', default=False, help='Do sample.')
    parser.add_argument('--num_kv_cache_blocks', type=int, default=4096, help='Number of KV cache blocks.')
    parser.add_argument('--num_max_batch_tokens', type=int, default=2048, help='Number of max batch tokens.')
    parser.add_argument('--num_max_batch_requests', type=int, default=32, help='Number of max batch requests.')
    parser.add_argument('--num_min_batch_requests', type=int, default=None, help='Number of min batch requests.')
    parser.add_argument('--max_context_len', type=int, default=4096, help='Number of max context length.')
    parser.add_argument('--use_ray_executor', action='store_true', default=False, help='Use ray executor.')
    parser.add_argument('--use_ray_worker', action='store_true', default=False, help='Use ray worker.')
    parser.add_argument('--use_chunked_prefill', action='store_true', default=False, help='Use chunked prefill.')
    parser.add_argument('--use_sparse_probs', action='store_true', default=False, help='Use sparse probs.')
    parser.add_argument('--use_cuda_graph', action='store_true', default=False, help='Use CUDA graph.')
    parser.add_argument('--cuda_graph_target_batch_sizes', nargs='+', type=int, default=[1, 2, 4, 8, 16], help='Batch sizes for processing.')
    parser.add_argument('--kv_cache_block_size', type=int, default=16, help='Number of KV cache block size.')
    parser.add_argument('--beta_threshold', type=float, default=0.9, help='Number of beta threshold.')
    parser.add_argument('--enable_nsight', action='store_true', default=False, help='Enable nsight.')
    parser.add_argument('--static_batch_profile', action='store_true', default=False, help='Enable static batch profile.')
    parser.add_argument('--run_profile_acceptance_prob', action='store_true', default=False, help='Run profile acceptance prob.')
    parser.add_argument('--acceptance_profile_path', type=str, default=None, help='Path to acceptance profile.')
    parser.add_argument('--use_only_beta_cutoff', action='store_true', default=False, help='Use only beta cutoff.')
    parser.add_argument('--use_data_parallel_draft', action='store_true', default=False, help='Use data parallel draft.')
    parser.add_argument('--draft_last_layer_idx', type=int, default=0, help='Number of draft last layer idx.')
    parser.add_argument('--verify_last_layer_idx', type=int, default=0, help='Number of verify last layer idx.')
    parser.add_argument('--profile_prompt_len', type=int, default=256, help='Number of profile prompt length.')
    parser.add_argument('--quantization', type=str, default=None, help='Quantization method')
    parser.add_argument('--no_cutoff', action='store_true', default=False, help='No cutoff.')
    parser.add_argument('--smart_spec_enabled', action='store_true', default=False, help='Smart spec enabled.')
    parser.add_argument('--moving_average_window', type=int, default=20, help='Moving average window for acceptance rate.')
    parser.add_argument('--svip_enabled', action='store_true', default=False, help='SVIP enabled.')
    parser.add_argument('--threshold_for_svip', type=float, default=0.5, help='Threshold for SVIP.')
    parser.add_argument('--eagle3_enabled', action='store_true', default=False, help='Eagle3 enabled.')
    parser.add_argument('--use_cache_offloading', action='store_true', default=False, help='Use cache offloading.')
    return parser

def start_nsys_profile():
    if NVTX_AVAILABLE or NVTX_ENV == "1":
        torch.cuda.cudart().cudaProfilerStart()

def stop_nsys_profile():
    if NVTX_AVAILABLE or NVTX_ENV == "1":
        torch.cuda.cudart().cudaProfilerStop()

def profile_nvtx(name=None):
    """Decorator that wraps a function with NVTX range markers for profiling.
    
    Args:
        name (str, optional): Name for the NVTX range. If None, uses function name.
    
    Returns:
        Decorated function that will be profiled with NVTX ranges if available.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not NVTX_AVAILABLE or NVTX_ENV == "0":
                return func(*args, **kwargs)
                
            range_name = name if name is not None else func.__name__
            nvtx.range_push(range_name)
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                nvtx.range_pop()
        return wrapper
    return decorator



def print_debug(function_name: str, **variables):
    global SCREEN_WIDTH
    
    # Find the longest name and type for alignment
    max_name_len = max(len(name) for name in variables)
    max_type_len = max(len(type(value).__name__) for value in variables.values())
    
    line_template = "          {name:<{name_len}} | {type:<{type_len}} : {value}"
    
    var_lines = [
        line_template.format(
            name=name,
            name_len=max_name_len,
            type=type(value).__name__,
            type_len=max_type_len,
            value=value
        )
        for name, value in variables.items()
    ]
    
    max_width = min(max(len(function_name), max(len(line) for line in var_lines)) + 20, SCREEN_WIDTH)
    
    dash_count = (max_width - len(function_name) - 2) // 2
    separator_part = "-" * dash_count
    top_separator = f"{separator_part} {function_name} {separator_part}"
    bottom_separator = "-" * max_width
    print("\n")
    print(top_separator)
    for line in var_lines:
        print(line)
    print(bottom_separator)

from transformers import AutoConfig
from huggingface_hub import hf_hub_download, snapshot_download

def check_and_download_model(model_name_or_path: str, download_path: str):
    print(f"Trying to download model '{model_name_or_path}' to '{download_path}'")
    os.makedirs(download_path, exist_ok=True)
    
    config = AutoConfig.from_pretrained(model_name_or_path, cache_dir=download_path, resume_download=True)
    
    essential_files = [
        "pytorch_model.bin",           # PyTorch 모델 파일
        "tokenizer.json",             # 토크나이저 파일
        "tokenizer_config.json",      # 토크나이저 구성 파일
        "vocab.txt",                  # BERT 계열 모델의 경우
        "merges.txt",                 # BPE 기반 토크나이저의 경우
        "special_tokens_map.json"      # 특별 토큰 맵 파일
    ]
    
    missing_files = []
    for file in essential_files:
        file_path = os.path.join(download_path, file)
        if not os.path.isfile(file_path):
            missing_files.append(file)
    
    if missing_files:
        print(f"모델 '{model_name_or_path}'의 다음 파일이 '{download_path}'에 존재하지 않습니다: {missing_files}")
        print(f"누락된 파일을 다운로드를 시작합니다.")
        hf_hub_download(repo_id=model_name_or_path, cache_dir=download_path, local_dir=download_path, force_download=False)
        print(f"'{file}' 파일이 '{download_path}'에 다운로드되었습니다.")
    else:
        print(f"모델 '{model_name_or_path}'의 모든 필수 파일이 '{download_path}'에 이미 존재합니다.")
    
    return download_path


def int_key_object_hook(obj):
    new_obj = {}
    for key, value in obj.items():
        try:
            new_key = int(key)
        except ValueError:
            new_key = key
        new_obj[new_key] = value
    return new_obj
