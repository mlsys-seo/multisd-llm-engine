clear
rm -rf /tmp/ray*
export TORCH_CUDA_ARCH_LIST="8.6"
export SDPROFILER_PROFILE_NVTX=1
export NCCL_P2P_LEVEL=NODE
export NCCL_P2P_DISABLE=1
export CUDA_LAUNCH_BLOCKING=0
export CUDA_VISIBLE_DEVICES=4
# export NCCL_DEBUG=INFO
export RAY_DEDUP_LOGS=0
export TORCH_USE_CUDA_DSA=0

MODEL_CACHE_DIR="/workspace/cache"

# nsys profile --trace=cuda,nvtx -o ./nsys/offline_inference_speculation --force-overwrite true \


NUM_DRAFT_STEPS=5
NUM_VERIFY_STEPS=1

MAX_PROMPT_LEN=512
MIN_NEW_TOKENS=64
MAX_NEW_TOKENS=512
# MIN_NEW_TOKENS=64
# MAX_NEW_TOKENS=128

NUM_MAX_BATCH_TOKENS=512
NUM_MAX_BATCH_REQUESTS=(16)
BETA_THRESHOLD=0.0
DATASET_NAME=("humanevalpack")
DATASET_SIZE=128
SEED=2025


MODEL_NAME_OR_PATH="Qwen/Qwen2.5-0.5B-Instruct"
DRAFT_MODEL_NAME_OR_PATH="facebook/layerskip-llama3.2-1B"
VERIFY_MODEL_NAME_OR_PATH="facebook/layerskip-llama3.2-1B"
TENSOR_PARALLEL_SIZE=1

    # --draft_model_name_or_path ${DRAFT_MODEL_NAME_OR_PATH} \
for dataset_name in ${DATASET_NAME[@]}; do
    ACCEPTANCE_PROFILE_PATH="/workspace/SDProfiler/examples/accepted_probs/CodeLlama-34b-Instruct-hf/humanevalpack_CodeLlama-34b-Instruct-hf.csv"

    for num_max_batch_requests in ${NUM_MAX_BATCH_REQUESTS[@]}; do
        echo "num_max_batch_requests: ${NUM_MAX_BATCH_REQUESTS}"

        # python offline_inference.py \
        #     --max_prompt_len ${MAX_PROMPT_LEN} \
        #     --model_cache_dir ${MODEL_CACHE_DIR} \
        #     --model_name_or_path ${MODEL_NAME_OR_PATH} \
        #     --tensor_parallel_size ${TENSOR_PARALLEL_SIZE} \
        #     --num_draft_steps ${NUM_DRAFT_STEPS} \
        #     --num_verify_steps ${NUM_VERIFY_STEPS} \
        #     --max_new_tokens ${MAX_NEW_TOKENS} \
        #     --min_new_tokens ${MIN_NEW_TOKENS} \
        #     --num_max_batch_tokens ${NUM_MAX_BATCH_TOKENS} \
        #     --num_max_batch_requests ${num_max_batch_requests} \
        #     --beta_threshold ${BETA_THRESHOLD} \
        #     --dataset_name ${dataset_name} \
        #     --dataset_size ${DATASET_SIZE} \
        #     --seed ${SEED} \
        #     --acceptance_profile_path ${ACCEPTANCE_PROFILE_PATH} \
        #     --use_chunked_prefill \
        #     --do_sample  


        DRAFT_LAST_LAYER_IDX=3
        VERIFY_LAST_LAYER_IDX=5
        python offline_inference.py \
            --max_prompt_len ${MAX_PROMPT_LEN} \
            --model_cache_dir ${MODEL_CACHE_DIR} \
            --model_name_or_path ${DRAFT_MODEL_NAME_OR_PATH} \
            --draft_model_name_or_path ${DRAFT_MODEL_NAME_OR_PATH} \
            --verify_model_name_or_path ${DRAFT_MODEL_NAME_OR_PATH} \
            --tensor_parallel_size ${TENSOR_PARALLEL_SIZE} \
            --num_draft_steps ${NUM_DRAFT_STEPS} \
            --num_verify_steps ${NUM_VERIFY_STEPS} \
            --max_new_tokens ${MAX_NEW_TOKENS} \
            --min_new_tokens ${MIN_NEW_TOKENS} \
            --num_max_batch_tokens ${NUM_MAX_BATCH_TOKENS} \
            --num_max_batch_requests ${num_max_batch_requests} \
            --beta_threshold ${BETA_THRESHOLD} \
            --dataset_name ${dataset_name} \
            --dataset_size ${DATASET_SIZE} \
            --seed ${SEED} \
            --draft_last_layer_idx ${DRAFT_LAST_LAYER_IDX} \
            --verify_last_layer_idx ${VERIFY_LAST_LAYER_IDX} \
            --acceptance_profile_path ${ACCEPTANCE_PROFILE_PATH} \
            --use_chunked_prefill \
            --do_sample \
            --static_batch_profile 
            

        # DRAFT_LAST_LAYER_IDX=7
        # VERIFY_LAST_LAYER_IDX=7
        # python offline_inference.py \
        #     --max_prompt_len ${MAX_PROMPT_LEN} \
        #     --model_cache_dir ${MODEL_CACHE_DIR} \
        #     --model_name_or_path ${MODEL_NAME_OR_PATH} \
        #     --draft_model_name_or_path ${MODEL_NAME_OR_PATH} \
        #     --tensor_parallel_size ${TENSOR_PARALLEL_SIZE} \
        #     --num_draft_steps ${NUM_DRAFT_STEPS} \
        #     --num_verify_steps ${NUM_VERIFY_STEPS} \
        #     --max_new_tokens ${MAX_NEW_TOKENS} \
        #     --min_new_tokens ${MIN_NEW_TOKENS} \
        #     --num_max_batch_tokens ${NUM_MAX_BATCH_TOKENS} \
        #     --num_max_batch_requests ${num_max_batch_requests} \
        #     --beta_threshold ${BETA_THRESHOLD} \
        #     --dataset_name ${dataset_name} \
        #     --dataset_size ${DATASET_SIZE} \
        #     --seed ${SEED} \
        #     --draft_last_layer_idx ${DRAFT_LAST_LAYER_IDX} \
        #     --verify_last_layer_idx ${VERIFY_LAST_LAYER_IDX} \
        #     --acceptance_profile_path ${ACCEPTANCE_PROFILE_PATH} \
        #     --use_chunked_prefill \
        #     --do_sample \
        #     --static_batch_profile \

    done
done