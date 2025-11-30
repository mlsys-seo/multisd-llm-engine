clear
rm -rf /tmp/ray*
export TORCH_CUDA_ARCH_LIST="8.0"
export SDPROFILER_PROFILE_NVTX=0
export NCCL_P2P_LEVEL=NODE
export NCCL_P2P_DISABLE=1
export CUDA_LAUNCH_BLOCKING=0
export CUDA_VISIBLE_DEVICES=0,1
# export NCCL_DEBUG=INFO
export RAY_DEDUP_LOGS=0

export TORCH_USE_CUDA_DSA=0
MODEL_CACHE_DIR="/workspace/cache"

MAX_PROMPT_LEN=256
MIN_NEW_TOKENS=64
MAX_NEW_TOKENS=3584

HF_MODEL_TOKEN=""

DATASET_NAME=("specbench--multiturn" "specbench--translation" "specbench--summarization" "specbench--qa" "specbench--math_reasoning" "specbench--rag")
# DATASET_NAME=("specbench--rag")

DATASET_SIZE=128
SEED=2025

NUM_DRAFT_STEPS=(5 7)
NUM_MAX_BATCH_TOKENS=512
NUM_MAX_BATCH_REQUESTS=(32 16 8)

MPS_PERCENTAGES=(30)
VERIFY_MPS_PERCENTAGES=95

MODEL_NAME_OR_PATH="Qwen/Qwen2.5-32B-Instruct"
VERIFY_MODEL_NAME_OR_PATH="Qwen/Qwen2.5-1.5B-Instruct"
DRAFT_MODEL_NAME_OR_PATH="Qwen/Qwen2.5-0.5B-Instruct"
TENSOR_PARALLEL_SIZE=2
ACCEPTANCE_PROFILE_PATH="/workspace/sdprofiler/examples/accepted_probs/Qwen2.5-32B-Instruct/sharegpt_Qwen2.5-32B-Instruct_Qwen2.5-1.5B-Instruct_Qwen2.5-0.5B-Instruct.csv"

        
DEFAULT_ARGS="--use_ray_worker"

DP_ARGS="--use_data_parallel_draft \
        --use_ray_worker \
        --parallelism_label DP"

for max_batch in ${NUM_MAX_BATCH_REQUESTS[@]}; do
        for dataset_name in ${DATASET_NAME[@]}; do

                BASE_ARGS="--model_cache_dir ${MODEL_CACHE_DIR} \
                        --tensor_parallel_size ${TENSOR_PARALLEL_SIZE} \
                        --seed ${SEED} \
                        --dataset_name ${dataset_name} \
                        --dataset_size ${DATASET_SIZE} \
                        --num_max_batch_tokens ${NUM_MAX_BATCH_TOKENS} \
                        --max_prompt_len ${MAX_PROMPT_LEN} \
                        --max_new_tokens ${MAX_NEW_TOKENS} \
                        --min_new_tokens ${MIN_NEW_TOKENS} \
                        --do_sample \
                        --static_batch_profile \
                        --use_chunked_prefill \
                        --use_cuda_graph \
                        --hf_model_token ${=None}"

                
                AR_ARGS="--model_name_or_path ${MODEL_NAME_OR_PATH} \
                        --num_max_batch_requests ${max_batch} \
                        --model_label AR"

                # # AR
                # python offline_inference.py \
                #         ${BASE_ARGS} \
                #         ${AR_ARGS} \
                #         ${DEFAULT_ARGS}

                for draft_steps in ${NUM_DRAFT_STEPS[@]}; do    

                        SD_ARGS="--model_name_or_path ${MODEL_NAME_OR_PATH} \
                                --draft_model_name_or_path ${DRAFT_MODEL_NAME_OR_PATH} \
                                --num_draft_steps ${draft_steps} \
                                --num_max_batch_requests ${max_batch}"

                        SV_ARGS="--model_name_or_path ${MODEL_NAME_OR_PATH} \
                                --draft_model_name_or_path ${DRAFT_MODEL_NAME_OR_PATH} \
                                --num_draft_steps ${draft_steps} \
                                --verify_model_name_or_path ${VERIFY_MODEL_NAME_OR_PATH} \
                                --num_verify_steps 1 \
                                --acceptance_profile_path ${ACCEPTANCE_PROFILE_PATH} \
                                --num_max_batch_requests ${max_batch} \
                                --model_label SV"

                        # SD
                        # python offline_inference.py \
                        #         ${BASE_ARGS} \
                        #         ${SD_ARGS} \
                        #         ${DEFAULT_ARGS} \
                        #         --model_label "SD"
                        
                        # # # SD + DP         
                        # # python offline_inference.py \
                        # #         ${BASE_ARGS} \
                        # #         ${SD_ARGS} \
                        # #         ${DP_ARGS} \
                        # #         --model_label "SD-DP"

                        # # # # SV
                        # # python offline_inference.py \
                        # #         ${BASE_ARGS} \
                        # #         ${SV_ARGS} \
                        # #         ${DEFAULT_ARGS}
                        
                        # SVIP
                        python offline_inference.py \
                                ${BASE_ARGS} \
                                ${SD_ARGS} \
                                --svip_enabled \
                                --threshold_for_svip 0.2 \
                                --model_label "SD-SVIP"

                        # # SMARTSPEC
                        python offline_inference.py \
                                ${BASE_ARGS} \
                                ${SD_ARGS} \
                                --smart_spec_enabled \
                                --moving_average_window 20 \
                                --model_label "SD-MA"


                        # Tierd
                        python offline_inference.py \
                                ${BASE_ARGS} \
                                ${SV_ARGS} \
                                ${DEFAULT_ARGS} \
                                --use_only_beta_cutoff \
                                --model_label "SV-Tierd"

                        # SV + DP
                        python offline_inference.py \
                                ${BASE_ARGS} \
                                ${SV_ARGS} \
                                ${DP_ARGS}



                done
        done
done