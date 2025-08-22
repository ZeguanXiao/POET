#!/bin/bash

export OPENAI_BASE_URL="https://api.deepseek.com/v1"
export OPENAI_API_KEY=""

res_ids=0,1,2,3  # GPU IDs

run_eval() {
    local gpu_id=$1
    shift
    CUDA_VISIBLE_DEVICES=$gpu_id python eval/alpacaeval2/run_eval.py "$@"
}

(
running=0
gpu_index=0

declare -a EVAL_COMMANDS=(
    # mistral-7b-base-dpo
    "--model_configs \"eval/alpacaeval2/configs/Mistral-7B-Base-SFT-SimPO.yaml\" --annotators_config weighted_alpaca_eval_deepseek_v3_0324 --output_path \"results/mistral-7b-base-dpo\" --override \"Mistral-7B-Base-SFT-SimPO.prompt_template=eval/alpacaeval2/templates/mistral_base.txt\" --override \"Mistral-7B-Base-SFT-SimPO.completions_kwargs.model_name=${ROOT_DIR}/DualPO/outputs/mistral-7b-base-dpo\" --override \"Mistral-7B-Base-SFT-SimPO.pretty_name=Mistral-7B-Base-DPO\""
    # mistral-7b-base-dpo-poet
    "--model_configs \"eval/alpacaeval2/configs/Mistral-7B-Base-SFT-SimPO.yaml\" --annotators_config weighted_alpaca_eval_deepseek_v3_0324 --output_path \"results/mistral-7b-base-dpo-poet\" --override \"Mistral-7B-Base-SFT-SimPO.prompt_template=eval/alpacaeval2/templates/mistral_base.txt\" --override \"Mistral-7B-Base-SFT-SimPO.completions_kwargs.model_name=${ROOT_DIR}/DualPO/outputs/mistral-7b-base-dpo-poet\" --override \"Mistral-7B-Base-SFT-SimPO.pretty_name=Mistral-7B-Base-DPO-POET\""
)

for cmd in "${EVAL_COMMANDS[@]}"
do
    while [ $running -ge ${#GPU_ARRAY[@]} ]; do
        wait -n
        running=$((running - 1))
    done

    current_gpu=${GPU_ARRAY[$gpu_index]}
    eval "run_eval $current_gpu $cmd" &
    
    running=$((running + 1))
    gpu_index=$(( (gpu_index + 1) % ${#GPU_ARRAY[@]} ))
done

wait
)












