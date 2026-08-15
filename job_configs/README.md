# LEAP Finetune Job Configs

These configs run with [Liquid4All/leap-finetune](https://github.com/Liquid4All/leap-finetune) —
Liquid AI's official finetuning framework with built-in evals, checkpointing, HF export, and GGUF export.

## Setup

```bash
git clone https://github.com/Liquid4All/leap-finetune.git ~/leap-finetune
cd ~/leap-finetune && uv sync
```

## Data format (LEAP "messages" style)

```jsonl
{"id": "1", "messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

Convert existing alpaca-style JSONL: `python3.13 scripts/convert_to_leap.py --input data/validated/sft_train.jsonl`

## Workflow

```bash
# 1. Train (SFT + LoRA) — 350M
uv run leap-finetune job_configs/leap_sft_350m.yaml

# 1b. Train (SFT + LoRA) — 1.2B
uv run leap-finetune job_configs/leap_sft_1_2b.yaml

# 2. Standalone Hindi/Hinglish eval against any checkpoint
uv run leap-finetune eval job_configs/leap_eval_standalone.yaml --output results.json

# 3. GGUF export from merged model (F16 + Q8_0 direct; Q4_K_M etc. need llama.cpp)
uv run leap-finetune export output/merged --quant F16 --quant Q8_0
```
