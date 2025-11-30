clear
rm -rf /tmp/ray*
export TORCH_CUDA_ARCH_LIST="8.0"
export SDPROFILER_PROFILE_NVTX=0
# export NCCL_P2P_LEVEL=NODE
# export NCCL_P2P_DISABLE=1
export CUDA_LAUNCH_BLOCKING=0
export CUDA_VISIBLE_DEVICES=0,1,2,3
# export NCCL_DEBUG=INFO
export RAY_DEDUP_LOGS=0

export TORCH_USE_CUDA_DSA=0

HF_MODEL_TOKEN=""

MODEL_CACHE_DIR="/workspace/cache"

MAX_PROMPT_LEN=512
MIN_NEW_TOKENS=512
MAX_NEW_TOKENS=1536

DATASET_NAME="sharegpt"
DATASET_SIZE=128
SEED=2025

NUM_DRAFT_STEPS=(5)
NUM_MAX_BATCH_TOKENS=512
NUM_MAX_BATCH_REQUESTS=(48 24)

MPS_PERCENTAGES=(30)
VERIFY_MPS_PERCENTAGES=95

MODEL_NAME_OR_PATH="Qwen/Qwen2.5-72B-Instruct"
VERIFY_MODEL_NAME_OR_PATH="Qwen/Qwen2.5-1.5B-Instruct"
DRAFT_MODEL_NAME_OR_PATH="Qwen/Qwen2.5-0.5B-Instruct"
TENSOR_PARALLEL_SIZE=4
ACCEPTANCE_PROFILE_PATH="./accepted_probs/Qwen2.5-72B-Instruct/sharegpt_Qwen2.5-72B-Instruct.csv"

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
        

DEFAULT_ARGS="--use_ray_worker"

DP_ARGS="--use_data_parallel_draft \
        --use_ray_worker \
        --parallelism_label DP"


for draft_steps in ${NUM_DRAFT_STEPS[@]}; do

        for max_batch in ${NUM_MAX_BATCH_REQUESTS[@]}; do
                export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=100
                
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

                
                # AR
                python offline_inference.py \
                        ${BASE_ARGS} \
                        ${AR_ARGS} \
                        ${DEFAULT_ARGS}

                # SD
                python offline_inference.py \
                        ${BASE_ARGS} \
                        ${SD_ARGS} \
                        ${DEFAULT_ARGS}
                    
                # # SD + DP         
                # python offline_inference.py \
                #         ${BASE_ARGS} \
                #         ${SD_ARGS} \
                #         ${DP_ARGS}

                # SV
                # python offline_inference.py \
                #         ${BASE_ARGS} \
                #         ${SV_ARGS} \
                #         ${DEFAULT_ARGS}
                
                # SV + DP
                python offline_inference.py \
                        ${BASE_ARGS} \
                        ${SV_ARGS} \
                        ${DP_ARGS}

        done
done
