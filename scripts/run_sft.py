#!/usr/bin/env python3
"""
SFT Trainer for LFM2.5-350M-Hindi using TRL + PEFT (LoRA).
Runs on GPU (Colab T4 / A100 recommended).
Usage: python3.13 scripts/run_sft.py --config configs/sft_config.yaml
"""

import argparse
import json
import logging
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)
from trl import SFTConfig, SFTTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_jsonl_dataset(path: str | Path) -> Dataset:
    """Load JSONL → HuggingFace Dataset."""
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return Dataset.from_list(data)


def format_chat_messages(examples: dict, tokenizer) -> dict:
    """Apply chat template to each example."""
    texts = []
    inputs = examples.get("input", [""] * len(examples["instruction"]))
    for inst, inp, out in zip(examples["instruction"], inputs, examples["output"]):
        msg = [
            {"role": "user", "content": f"{inst} {inp}".strip()},
            {"role": "assistant", "content": out},
        ]
        texts.append(tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=False))
    return {"text": texts}


def build_lora_config(cfg: dict) -> LoraConfig:
    return LoraConfig(
        r=cfg.get("lora_r", 16),
        lora_alpha=cfg.get("lora_alpha", 32),
        lora_dropout=cfg.get("lora_dropout", 0.05),
        task_type=TaskType.CAUSAL_LM,
        target_modules=cfg.get("target_modules", [
            "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.out_proj",
            "feed_forward.w1", "feed_forward.w2", "feed_forward.w3",
            "conv.in_proj", "conv.out_proj",
        ]),
    )


def build_sft_config(cfg: dict) -> SFTConfig:
    """Build SFTConfig from YAML cfg, only passing supported keys."""
    supported = {
        "output_dir", "num_train_epochs", "per_device_train_batch_size",
        "gradient_accumulation_steps", "learning_rate", "lr_scheduler_type",
        "warmup_ratio", "weight_decay", "max_length", "packing",
        "bf16", "fp16", "logging_steps", "save_strategy", "save_total_limit",
        "save_steps", "eval_strategy", "report_to", "seed",
        "gradient_checkpointing", "gradient_checkpointing_kwargs",
    }
    kwargs = {k: v for k, v in cfg.items() if k in supported}
    kwargs["output_dir"] = str(Path(cfg["output_dir"]) / "training")
    kwargs["dataset_text_field"] = "text"
    kwargs["max_length"] = cfg.get("max_seq_length", 1024)
    kwargs["gradient_checkpointing"] = True
    kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    kwargs["remove_unused_columns"] = False
    return SFTConfig(**kwargs)


def run_sft(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))

    logger.info("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model_name_or_path"], trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Optionally load quantized (QLoRA) for memory efficiency
    quant_config = None
    if cfg.get("qlora", False):
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if cfg.get("bf16") else torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name_or_path"],
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if cfg.get("bf16") else torch.float16,
        quantization_config=quant_config,
    )
    model.config.use_cache = False  # required for gradient checkpointing

    logger.info("Applying LoRA...")
    lora_cfg = build_lora_config(cfg)
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    logger.info("Loading datasets...")
    train_ds = load_jsonl_dataset(cfg["dataset_paths"][0])
    train_ds = train_ds.select(range(min(len(train_ds), cfg.get("max_train_samples", 15000))))
    # Drop only columns that actually exist (some sources lack "language")
    drop_cols = [c for c in ("instruction", "input", "output", "language") if c in train_ds.column_names]
    train_ds = train_ds.map(
        lambda ex: format_chat_messages(ex, tokenizer),
        batched=True,
        remove_columns=drop_cols,
    )

    eval_ds = None
    if cfg.get("eval_dataset_path"):
        eval_ds = load_jsonl_dataset(cfg["eval_dataset_path"])
        eval_drop = [c for c in ("instruction", "input", "output", "language") if c in eval_ds.column_names]
        eval_ds = eval_ds.map(
            lambda ex: format_chat_messages(ex, tokenizer),
            batched=True,
            remove_columns=eval_drop,
        )

    sft_cfg = build_sft_config(cfg)

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    logger.info("Starting SFT training...")
    trainer.train()

    # Save LoRA adapter
    adapter_dir = Path(cfg["output_dir"]) / "adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    logger.info(f"✅ LoRA adapter saved to {adapter_dir}")

    # Merge and save full model
    merged = model.merge_and_unload()
    merged_dir = Path(cfg["output_dir"]) / "merged"
    merged.save_pretrained(str(merged_dir))
    tokenizer.save_pretrained(str(merged_dir))
    logger.info(f"✅ Merged model saved to {merged_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/sft_config.yaml")
    args = parser.parse_args()
    run_sft(args.config)
