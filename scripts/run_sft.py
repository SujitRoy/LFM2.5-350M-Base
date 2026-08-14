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
    TrainingArguments,
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
    for inst, inp, out in zip(
        examples["instruction"], examples.get("input", [""] * len(examples["instruction"])),
        examples["output"]
    ):
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
        target_modules=cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
    )


def build_training_args(cfg: dict) -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(Path(cfg["output_dir"]) / "training"),
        num_train_epochs=cfg.get("num_train_epochs", 3),
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 4),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 8),
        learning_rate=cfg.get("learning_rate", 2e-4),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=cfg.get("warmup_ratio", 0.05),
        weight_decay=cfg.get("weight_decay", 0.01),
        max_seq_length=cfg.get("max_seq_length", 1024),
        packing=cfg.get("packing", False),
        bf16=cfg.get("bf16", True),
        fp16=cfg.get("fp16", False),
        logging_steps=cfg.get("logging_steps", 20),
        save_strategy=cfg.get("save_strategy", "steps"),
        save_total_limit=cfg.get("save_total_limit", 3),
        save_steps=cfg.get("save_steps", 500),
        eval_strategy=cfg.get("eval_strategy", "epoch"),
        report_to=cfg.get("report_to", "tensorboard"),
        seed=cfg.get("seed", 42),
        remove_unused_columns=False,
    )


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

    # Enable gradient checkpointing for memory savings
    model.gradient_checkpointing_enable()

    logger.info("Applying LoRA...")
    lora_cfg = build_lora_config(cfg)
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    logger.info("Loading datasets...")
    train_ds = load_jsonl_dataset(cfg["dataset_paths"][0])
    train_ds = train_ds.select(range(min(len(train_ds), 15000)))  # cap for speed
    train_ds = train_ds.map(
        lambda ex: format_chat_messages(ex, tokenizer),
        batched=True,
        remove_columns=["instruction", "input", "output", "language"],
    )

    eval_ds = None
    if cfg.get("eval_dataset_path"):
        eval_ds = load_jsonl_dataset(cfg["eval_dataset_path"])
        eval_ds = eval_ds.map(
            lambda ex: format_chat_messages(ex, tokenizer),
            batched=True,
            remove_columns=["instruction", "input", "output", "language"],
        )

    sft_cfg = SFTConfig(
        **{k: v for k, v in cfg.items() if k not in {
            "model_name_or_path", "dataset_paths", "eval_dataset_path",
            "lora_r", "lora_alpha", "lora_dropout", "target_modules",
        }},
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
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
