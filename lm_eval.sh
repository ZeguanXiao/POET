#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3

export PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $PORT"

arc_challenge(){
    MODEL=$1
    accelerate launch --main_process_port $PORT -m lm_eval --model hf \
    --model_args "pretrained=$MODEL,dtype=bfloat16" \
    --tasks arc_challenge \
    --num_fewshot 25 \
    --batch_size 2 \
    --log_samples \
    --output_path results/open_llm/arc_challenge
}

hellaswag(){
    MODEL=$1
    accelerate launch --main_process_port $PORT -m lm_eval --model hf \
    --model_args "pretrained=$MODEL,dtype=bfloat16" \
    --tasks hellaswag \
    --num_fewshot 10 \
    --batch_size 2 \
    --log_samples \
    --output_path results/open_llm/hellaswag
}

truthfulqa(){
    MODEL=$1
    accelerate launch --main_process_port $PORT -m lm_eval --model hf \
    --model_args "pretrained=$MODEL,dtype=bfloat16" \
    --tasks truthfulqa \
    --num_fewshot 0 \
    --batch_size 2 \
    --log_samples \
    --output_path results/open_llm/truthfulqa
}

mmlu(){
    MODEL=$1
    accelerate launch --main_process_port $PORT -m lm_eval --model hf \
    --model_args "pretrained=$MODEL,dtype=bfloat16" \
    --tasks mmlu \
    --num_fewshot 5 \
    --batch_size 2 \
    --log_samples \
    --output_path results/open_llm/mmlu
}

winogrande(){
    MODEL=$1
    accelerate launch --main_process_port $PORT -m lm_eval --model hf \
    --model_args "pretrained=$MODEL,dtype=bfloat16" \
    --tasks winogrande \
    --num_fewshot 5 \
    --batch_size 2 \
    --log_samples \
    --output_path results/open_llm/winogrande
}

gsm8k(){
    MODEL=$1
    accelerate launch --main_process_port $PORT -m lm_eval --model hf \
    --model_args "pretrained=$MODEL,dtype=bfloat16" \
    --tasks gsm8k \
    --num_fewshot 5 \
    --batch_size 2 \
    --log_samples \
    --output_path results/open_llm/gsm8k
}

MODEL_LIST=(
    "outputs/mistral-7b-base-dpo"
    "outputs/mistral-7b-base-dpo-poet"
)

for MODEL in ${MODEL_LIST[@]}
do
    echo "Running model: $MODEL"

    arc_challenge $MODEL
    hellaswag $MODEL
    truthfulqa $MODEL
    mmlu $MODEL
    winogrande $MODEL
    gsm8k $MODEL
done


