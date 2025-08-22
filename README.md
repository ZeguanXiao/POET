# POET: Prefix-Oriented Equal-length Training

This repository contains the code of our paper “Towards Bridging the Reward-Generation Gap in Direct Alignment Algorithms” ([arXiv:2506.09457](https://arxiv.org/abs/2506.09457)). We introduce POET (Prefix-Oriented Equal-length Training), a simple plug-in for Direct Alignment Algorithms (DAAs) such as DPO and SimPO. POET truncates both preferred and dispreferred responses in each pair to the shorter length, implicitly encouraging optimization to converge across all positions and pay more attention to prefix tokens. We show consistent gains on AlpacaEval 2 and improvements on downstream tasks when applying POET to DPO and SimPO.

## 🔗 Quick Links
- Paper: https://arxiv.org/abs/2506.09457
- Training configs: `training_configs/*-poet.yaml`

## 🧰 Installation
For environment installation, please see and run `build_env.sh`:

```bash
bash build_env.sh
```


## 🚀 Training (DPO/SimPO + POET)
We provide ready-to-run configs with POET enabled across popular models. Launch with Accelerate + DeepSpeed ZeRO-3 and pick a POET config:


- DPO + Mistral-7B-Base (POET):
```bash
ACCELERATE_LOG_LEVEL=info accelerate launch \
  --config_file accelerate_configs/deepspeed_zero3.yaml \
  scripts/run_simpo.py training_configs/mistral-7b-base-dpo-poet.yaml \
  --output_dir=outputs/mistral-7b-base-dpo-poet \
  --run_name=mistral-7b-base-dpo-poet
```

- SimPO + Mistral-7B-Base (POET):
```bash
ACCELERATE_LOG_LEVEL=info accelerate launch \
  --config_file accelerate_configs/deepspeed_zero3.yaml \
  scripts/run_simpo.py training_configs/mistral-7b-base-simpo-poet.yaml \
  --output_dir=outputs/mistral-7b-base-simpo-poet \
  --run_name=mistral-7b-base-simpo-poet
```

Other provided POET configs:
- `llama-3-8b-base-dpo-poet.yaml`, `llama-3-8b-base-simpo-poet.yaml`
- `llama-3-8b-instruct-dpo-v2-poet.yaml`, `llama-3-8b-instruct-simpo-v2-poet.yaml`
- `gemma-2-9b-it-dpo-poet.yaml`, `gemma-2-9b-it-simpo-poet.yaml`

Enable POET by setting `use_poet: true` in the training config.


## 📊 Evaluation

- AlpacaEval 2: We provide helpers for AlpacaEval 2; see `alpacaeval.sh` and `eval/alpacaeval2/run_eval.py`. Requires setting an API key for the chosen annotator (e.g., `OPENAI_API_KEY`/DeepSeek endpoint) inside `alpacaeval.sh`.

- Arena-Hard: Please refer to to the [Arena-Hard-Auto repo](https://github.com/lm-sys/arena-hard-auto) for evaluation.

- Open LLM Leaderboard tasks (ARC, HellaSwag, MMLU, GSM8K, ...). See `lm_eval.sh` (uses EleutherAI/lm-evaluation-harness via Accelerate).


## 🙌 Acknowledgements
This codebase builds upon and adapts components from the excellent SimPO and alignment-handbook projects.

## 📚 Citation
If you find POET helpful, please cite:

```bibtex
@article{xiao2025poet,
  title   = {Towards Bridging the Reward-Generation Gap in Direct Alignment Algorithms},
  author  = {Xiao, Zeguan and Chen, Yun and Chen, Guanhua and Ke, Tang},
  journal = {arXiv preprint arXiv:2506.09457},
  year    = {2025}
}
```
