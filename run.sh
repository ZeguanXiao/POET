#!/bin/bash


export CUDA_VISIBLE_DEVICES=0,1,2,3

export WANDB_PROJECT="poet"

# Without POET
ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file accelerate_configs/deepspeed_zero3.yaml \
    scripts/run_simpo.py training_configs/mistral-7b-base-dpo.yaml \
    --output_dir=outputs/mistral-7b-base-dpo \
    --run_name=mistral-7b-base-dpo

# With POET
ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file accelerate_configs/deepspeed_zero3.yaml \
    scripts/run_simpo.py training_configs/mistral-7b-base-dpo-poet.yaml \
    --output_dir=outputs/mistral-7b-base-dpo-poet \
    --run_name=mistral-7b-base-dpo-poet


