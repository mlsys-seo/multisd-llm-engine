clear
rm -rf /tmp/ray*
export TORCH_CUDA_ARCH_LIST="8.6"
export SDPROFILER_PROFILE_NVTX=1
export NCCL_P2P_LEVEL=NODE
export NCCL_P2P_DISABLE=1
export CUDA_LAUNCH_BLOCKING=0
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export NCCL_DEBUG=INFO
export RAY_DEDUP_LOGS=0

export TORCH_USE_CUDA_DSA=0

MODEL_CACHE_DIR="/workspace/cache"

# nsys profile --trace=cuda,nvtx -o ./nsys/offline_inference_speculation --force-overwrite true \


NUM_DRAFT_STEPS=5
NUM_VERIFY_STEPS=1

MAX_PROMPT_LEN=512
MIN_NEW_TOKENS=16
MAX_NEW_TOKENS=16

NUM_MAX_BATCH_TOKENS=512
NUM_MAX_BATCH_REQUESTS=64
BETA_THRESHOLD=0.0
DATASET_NAME="sharegpt"
DATASET_SIZE=64
SEED=2025

# --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" \
# --draft_model "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" \
# --verify_model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \

# MODEL_NAME_OR_PATH="Qwen/Qwen2.5-72B-Instruct"
# VERIFY_MODEL_NAME_OR_PATH="Qwen/Qwen2.5-1.5B-Instruct"
# DRAFT_MODEL_NAME_OR_PATH="Qwen/Qwen2.5-0.5B-Instruct"
# TENSOR_PARALLEL_SIZE=8

# MODEL_NAME_OR_PATH="Qwen/Qwen2.5-72B-Instruct"
# VERIFY_MODEL_NAME_OR_PATH="Qwen/Qwen2.5-1.5B-Instruct"
# DRAFT_MODEL_NAME_OR_PATH="Qwen/Qwen2.5-0.5B-Instruct"
# TENSOR_PARALLEL_SIZE=8
# ACCEPTANCE_PROFILE_PATH="./accepted_probs/Qwen2.5-72B-Instruct/sharegpt_Qwen2.5-72B-Instruct.csv"

MODEL_NAME_OR_PATH="Qwen/Qwen2.5-3B-Instruct"
VERIFY_MODEL_NAME_OR_PATH="Qwen/Qwen2.5-1.5B-Instruct"
DRAFT_MODEL_NAME_OR_PATH="Qwen/Qwen2.5-0.5B-Instruct"
TENSOR_PARALLEL_SIZE=1
ACCEPTANCE_PROFILE_PATH="./accepted_probs/Qwen2.5-3B-Instruct/sharegpt_Qwen2.5-3B-Instruct.csv"

# MODEL_NAME_OR_PATH="EleutherAI/pythia-12b"
# VERIFY_MODEL_NAME_OR_PATH="EleutherAI/pythia-410m"
# DRAFT_MODEL_NAME_OR_PATH="EleutherAI/pythia-160m"
# TENSOR_PARALLEL_SIZE=2
# ACCEPTANCE_PROFILE_PATH="./accepted_probs/pythia-12b/sharegpt_pythia-12b.csv"

# nsys profile --trace=cuda,nvtx -o ./nsys/offline_inference_3speculation --force-overwrite true \

mkdir -p ./nsys

nsys profile --trace=cuda,nvtx --capture-range=cudaProfilerApi --capture-range-end=stop-shutdown --wait=primary \
    -o ./nsys/new_offline_inference_3speculation_${NUM_MAX_BATCH_REQUESTS} --force-overwrite true \
        python offline_inference.py \
            --max_prompt_len ${MAX_PROMPT_LEN} \
            --model_cache_dir ${MODEL_CACHE_DIR} \
            --model_name_or_path ${MODEL_NAME_OR_PATH} \
            --draft_model_name_or_path ${DRAFT_MODEL_NAME_OR_PATH} \
            --verify_model_name_or_path ${VERIFY_MODEL_NAME_OR_PATH} \
            --tensor_parallel_size ${TENSOR_PARALLEL_SIZE} \
            --num_draft_steps ${NUM_DRAFT_STEPS} \
            --num_verify_steps ${NUM_VERIFY_STEPS} \
            --max_new_tokens ${MAX_NEW_TOKENS} \
            --min_new_tokens ${MIN_NEW_TOKENS} \
            --num_max_batch_tokens ${NUM_MAX_BATCH_TOKENS} \
            --num_max_batch_requests ${NUM_MAX_BATCH_REQUESTS} \
            --beta_threshold ${BETA_THRESHOLD} \
            --dataset_name ${DATASET_NAME} \
            --dataset_size ${DATASET_SIZE} \
            --seed ${SEED} \
            --use_cuda_graph \
            --acceptance_profile_path ${ACCEPTANCE_PROFILE_PATH}