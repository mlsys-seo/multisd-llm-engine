clear
rm -rf /tmp/ray*
export TORCH_CUDA_ARCH_LIST="8.6"
export SDPROFILER_PROFILE_NVTX=0
# export NCCL_P2P_LEVEL=NODE
# export NCCL_P2P_DISABLE=1
export CUDA_LAUNCH_BLOCKING=0
export CUDA_VISIBLE_DEVICES=0
# export NCCL_DEBUG=INFO
export RAY_DEDUP_LOGS=0

export TORCH_USE_CUDA_DSA=0

HF_MODEL_TOKEN=""

MODEL_CACHE_DIR="/data/models"

MAX_PROMPT_LEN=512
MIN_NEW_TOKENS=64
MAX_NEW_TOKENS=512

DATASET_NAME="specbench"
DATASET_SIZE=128
SEED=2025

NUM_DRAFT_STEPS=(5)
NUM_MAX_BATCH_TOKENS=512
NUM_MAX_BATCH_REQUESTS=(1 8 16 24 32)

MODEL_NAME_OR_PATH="meta-llama/Llama-2-13b-chat-hf"
VERIFY_MODEL_NAME_OR_PATH="Felladrin/Llama-160M-Chat-v1"
DRAFT_MODEL_NAME_OR_PATH="Felladrin/Llama-68M-Chat-v1"
TENSOR_PARALLEL_SIZE=2
ACCEPTANCE_PROFILE_PATH="./accepted_probs/Llama-2-13b-chat-hf/specbench_Llama-2-13b-chat-hf_Llama-160M-Chat-v1_Llama-68M-Chat-v1.csv"

BASE_ARGS="--model_cache_dir ${MODEL_CACHE_DIR} \
        --tensor_parallel_size ${TENSOR_PARALLEL_SIZE} \
        --seed ${SEED} \
        --dataset_name ${DATASET_NAME} \
        --dataset_size ${DATASET_SIZE} \
        --num_max_batch_tokens ${NUM_MAX_BATCH_TOKENS} \
        --max_prompt_len ${MAX_PROMPT_LEN} \
        --max_new_tokens ${MAX_NEW_TOKENS} \
        --min_new_tokens ${MIN_NEW_TOKENS} \
        --do_sample \
        --static_batch_profile \
        --use_chunked_prefill \
        --use_cuda_graph \
        --use_ray_worker"
        


AR_ARGS="--model_name_or_path ${MODEL_NAME_OR_PATH} \
        --num_max_batch_requests ${NUM_MAX_BATCH_REQUESTS} \
        --model_label AR"


SD_ARGS="--model_name_or_path ${MODEL_NAME_OR_PATH} \
        --draft_model_name_or_path ${DRAFT_MODEL_NAME_OR_PATH} \
        --num_draft_steps ${NUM_DRAFT_STEPS} \
        --num_max_batch_requests ${NUM_MAX_BATCH_REQUESTS} \
        --model_label SD"


# python smart_spec_inference.py \
#         ${BASE_ARGS} \
#         ${AR_ARGS}

# SD
python smart_spec_inference.py \
        ${BASE_ARGS} \
        ${SD_ARGS} \
        --moving_average_window 20
