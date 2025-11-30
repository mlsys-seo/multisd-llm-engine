clear
rm -rf /tmp/ray*
export TORCH_CUDA_ARCH_LIST="8.6"
export SDPROFILER_PROFILE_NVTX=0
export NCCL_P2P_LEVEL=NODE
export NCCL_P2P_DISABLE=1
export CUDA_LAUNCH_BLOCKING=0
export CUDA_VISIBLE_DEVICES=0
# export NCCL_DEBUG=INFO
export RAY_DEDUP_LOGS=0

export TORCH_USE_CUDA_DSA=0

HF_MODEL_TOKEN=""

MODEL_CACHE_DIR="/workspace/cache"
# MODEL_CACHE_DIR="/home/work/.models"

MAX_PROMPT_LEN=1024
MIN_NEW_TOKENS=256
MAX_NEW_TOKENS=1024

DATASET_NAME="specbench"
DATASET_SIZE=32
SEED=2025

NUM_DRAFT_STEPS=(5)
NUM_MAX_BATCH_TOKENS=512
NUM_MAX_BATCH_REQUESTS=(1)

MPS_PERCENTAGES=(30)
VERIFY_MPS_PERCENTAGES=95

MODEL_NAME_OR_PATH="Qwen/Qwen2.5-14B-Instruct"
VERIFY_MODEL_NAME_OR_PATH="Qwen/Qwen2.5-1.5B-Instruct"
DRAFT_MODEL_NAME_OR_PATH="Qwen/Qwen2.5-0.5B-Instruct"
TENSOR_PARALLEL_SIZE=1
ACCEPTANCE_PROFILE_PATH="./accepted_probs/Qwen2.5-14B-Instruct/sharegpt_Qwen2.5-14B-Instruct_Qwen2.5-1.5B-Instruct_Qwen2.5-0.5B-Instruct.csv"


# MODEL_NAME_OR_PATH="meta-llama/CodeLlama-34b-hf"
# VERIFY_MODEL_NAME_OR_PATH="TinyLlama/TinyLlama_v1.1_math_code"
# DRAFT_MODEL_NAME_OR_PATH="amd/AMD-Llama-135m-code"
# TENSOR_PARALLEL_SIZE=1
# ACCEPTANCE_PROFILE_PATH="./accepted_probs/CodeLlama-34b-hf/mbpp_CodeLlama-34b-hf_TinyLlama_v1.1_math_code_AMD-Llama-135m-code.csv"

# MODEL_NAME_OR_PATH="Qwen/Qwen2.5-32B-Instruct"
# VERIFY_MODEL_NAME_OR_PATH="Qwen/Qwen2.5-1.5B-Instruct"
# DRAFT_MODEL_NAME_OR_PATH="Qwen/Qwen2.5-0.5B-Instruct"
# TENSOR_PARALLEL_SIZE=1
# ACCEPTANCE_PROFILE_PATH="./accepted_probs/Qwen2.5-32B-Instruct/sharegpt_Qwen2.5-32B-Instruct_Qwen2.5-1.5B-Instruct_Qwen2.5-0.5B-Instruct.csv"

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
        --use_cuda_graph
        --hf_model_token ${HF_MODEL_TOKEN}"
        

DEFAULT_ARGS="--use_ray_worker"

DP_ARGS="--use_data_parallel_draft \
        --use_ray_worker \
        --parallelism_label DP"

for max_batch in ${NUM_MAX_BATCH_REQUESTS[@]}; do
        for dataset_name in ${DATASET_NAME[@]}; do
                for draft_steps in ${NUM_DRAFT_STEPS[@]}; do    

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
                                --hf_model_token ${HF_MODEL_TOKEN}"
                        
                        AR_ARGS="--model_name_or_path ${MODEL_NAME_OR_PATH} \
                                --num_max_batch_requests ${max_batch} \
                                --model_label AR"

                        SD_ARGS="--model_name_or_path ${MODEL_NAME_OR_PATH} \
                                --draft_model_name_or_path ${DRAFT_MODEL_NAME_OR_PATH} \
                                --num_draft_steps ${draft_steps} \
                                --num_max_batch_requests ${max_batch} \
                                --model_label SD"

                        SV_ARGS="--model_name_or_path ${MODEL_NAME_OR_PATH} \
                                --draft_model_name_or_path ${DRAFT_MODEL_NAME_OR_PATH} \
                                --num_draft_steps ${draft_steps} \
                                --verify_model_name_or_path ${VERIFY_MODEL_NAME_OR_PATH} \
                                --num_verify_steps 1 \
                                --acceptance_profile_path ${ACCEPTANCE_PROFILE_PATH} \
                                --num_max_batch_requests ${max_batch} \
                                --model_label SV"

                        # # AR
                        # python offline_inference.py \
                        #         ${BASE_ARGS} \
                        #         ${AR_ARGS} \
                        #         ${DEFAULT_ARGS}


                        # SD
                        python offline_inference.py \
                                ${BASE_ARGS} \
                                ${SD_ARGS} \
                                ${DEFAULT_ARGS}



                done
        done
done