clear
rm -rf /tmp/ray*
export TORCH_CUDA_ARCH_LIST="8.6"
export SDPROFILER_PROFILE_NVTX=0
export NCCL_P2P_LEVEL=NODE
export NCCL_P2P_DISABLE=1
export CUDA_LAUNCH_BLOCKING=0
export CUDA_VISIBLE_DEVICES=0,1
# export NCCL_DEBUG=INFO
export RAY_DEDUP_LOGS=0

export TORCH_USE_CUDA_DSA=0

HF_MODEL_TOKEN=""

MODEL_CACHE_DIR="/workspace/cache"

MAX_CONTEXT_LEN=1024

DATASET_NAME="specbench"
DATASET_SIZE=32
SEED=2025

NUM_DRAFT_STEPS=5
NUM_MAX_BATCH_TOKENS=512
NUM_MAX_BATCH_REQUESTS=1

MODEL_NAME_OR_PATH="Qwen/Qwen2.5-32B-Instruct"
DRAFT_MODEL_NAME_OR_PATH="Qwen/Qwen2.5-0.5B-Instruct"
TENSOR_PARALLEL_SIZE=2

BASE_ARGS="--model_cache_dir ${MODEL_CACHE_DIR} \
        --tensor_parallel_size ${TENSOR_PARALLEL_SIZE} \
        --seed ${SEED} \
        --dataset_name ${DATASET_NAME} \
        --dataset_size ${DATASET_SIZE} \
        --num_max_batch_tokens ${NUM_MAX_BATCH_TOKENS} \
        --max_context_len ${MAX_CONTEXT_LEN} \
        --do_sample \
        --static_batch_profile \
        --use_chunked_prefill \
        --use_cuda_graph \
        --use_ray_worker \
        --hf_model_token ${HF_MODEL_TOKEN}"


        
SD_ARGS="--model_name_or_path ${MODEL_NAME_OR_PATH} \
        --draft_model_name_or_path ${DRAFT_MODEL_NAME_OR_PATH} \
        --num_draft_steps ${NUM_DRAFT_STEPS} \
        --num_max_batch_requests ${NUM_MAX_BATCH_REQUESTS}"

python offline_inference.py \
        ${BASE_ARGS} \
        ${SD_ARGS} \
        --model_label "SD"
