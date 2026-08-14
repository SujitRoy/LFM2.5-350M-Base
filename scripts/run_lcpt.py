#!/usr/bin/env python3
"""
Continued Pretraining (LCPT) for LFM2.5-350M on Hindi/Hinglish text.
Usage: python3.13 scripts/run_lcpt.py --config configs/lcpt_config.yaml
"""

import argparse
import logging
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_text_corpus(path: str | Path) -> list[str]:
    """Load newline-separated text lines into a Dataset."""
    with open(path, encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    return Dataset.from_dict({"text": lines})


def chunk_text(dataset: Dataset, max_len: int) -> Dataset:
    """Chunk long texts into max_len-token segments."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("LiquidAI/LFM2.5-350M-Base")
    texts = []
    for doc in dataset["text"]:
        encoded = tok.encode(doc, truncation=False)
        for i in range(0, len(encoded), max_len):
            chunk = encoded[i:i + max_len]
            if len(chunk) >= 64:  # minimum useful chunk
                texts.append(tok.decode(chunk))
    return Dataset.from_dict({"text": texts})


def run_lcpt(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg.get("seed", 42))

    logger.info("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name_or_path"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name_or_path"],
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    logger.info("Loading corpus...")
    ds = load_text_corpus(cfg["dataset_path"])
    ds = chunk_text(ds, cfg.get("max_seq_length", 1024))
    logger.info(f"Corpus: {len(ds)} chunks")

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=cfg.get("max_seq_length", 1024),
            padding=False,
        )

    ds = ds.map(tokenize_fn, batched=True, remove_columns=["text"])

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    train_cfg = TrainingArguments(
        output_dir=str(Path(cfg["output_dir"]) / "training"),
        num_train_epochs=cfg.get("num_train_epochs", 2),
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 8),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 4),
        learning_rate=cfg.get("learning_rate", 2e-4),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=cfg.get("warmup_ratio", 0.05),
        weight_decay=cfg.get("weight_decay", 0.01),
        bf16=cfg.get("bf16", True),
        logging_steps=cfg.get("logging_steps", 50),
        save_strategy=cfg.get("save_strategy", "epoch"),
        report_to=cfg.get("report_to", "tensorboard"),
        seed=cfg.get("seed", 42),
    )

    trainer = Trainer(
        model=model,
        args=train_cfg,
        train_dataset=ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    logger.info("Starting LCPT training...")
    trainer.train()

    out_dir = Path(cfg["output_dir"]) / "lcpt_merged"
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    logger.info(f"✅ LCPT model saved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/lcpt_config.yaml")
    args = parser.parse_args()
    run_lcpt(args.config)
