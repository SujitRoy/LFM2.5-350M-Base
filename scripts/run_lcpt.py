#!/usr/bin/env python3
"""
Continued Pretraining (LCPT) for LFM2.5 on Hindi/Hinglish text.
Supports full fine-tune (350M) or LoRA (350M / 1.2B, for T4-class GPUs).
Usage: python3.13 scripts/run_lcpt.py --config configs/lcpt_config.yaml
"""

import argparse
import logging
import os
from pathlib import Path

# Must be set before the first CUDA allocation: reduces fragmentation on 16GB T4s
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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


def load_text_corpus(path: str | Path, max_stories: int | None = None, seed: int = 42) -> Dataset:
    """Load newline-separated text lines into a Dataset.

    Memory-aware: streams the file, reservoir-samples when max_stories is set,
    so we never hold the full corpus in RAM.
    """
    import random as _random

    rng = _random.Random(seed)
    sample: list[str] = []
    n_seen = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_seen += 1
            if max_stories is None:
                sample.append(line)          # no cap: grow (caller beware)
            elif len(sample) < max_stories:
                sample.append(line)
            else:
                j = rng.randrange(n_seen)     # reservoir sampling: uniform sample
                if j < max_stories:
                    sample[j] = line
    logger.info(f"Corpus: kept {len(sample):,} of {n_seen:,} lines")
    return Dataset.from_dict({"text": sample})


def chunk_text(dataset: Dataset, tokenizer, max_len: int, batch_size: int = 512) -> Dataset:
    """Chunk long texts into max_len-token segments. Batched: uses the Rust
    tokenizer's encode_batch (multi-threaded) instead of a per-text Python loop.
    """
    rust = getattr(tokenizer, "_tokenizer", None) or tokenizer
    texts_out: list[str] = []
    buf: list[str] = []

    def flush():
        if not buf:
            return
        encs = rust.encode_batch(buf)
        for e in encs:
            ids = e.ids if hasattr(e, "ids") else e["input_ids"]
            for i in range(0, len(ids), max_len):
                chunk = ids[i : i + max_len]
                if len(chunk) >= 64:  # minimum useful chunk
                    texts_out.append(rust.decode(chunk))
        buf.clear()

    for doc in dataset["text"]:
        buf.append(doc)
        if len(buf) >= batch_size:
            flush()
    flush()
    return Dataset.from_dict({"text": texts_out})


def run_lcpt(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg.get("seed", 42))

    model_name = cfg["model_name_or_path"]
    max_len = cfg.get("max_seq_length", 1024)

    logger.info(f"Loading model and tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    use_bf16 = cfg.get("bf16", True)
    use_fp16 = cfg.get("fp16", False)
    # AMP rule: mixed-precision training requires fp32 master weights.
    # bf16 needs no GradScaler, so bf16 loading is safe; fp16 AMP must load fp32.
    if use_fp16:
        dtype = torch.float32
    elif use_bf16:
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
    )

    # Optional LoRA for memory-constrained LCPT (recommended for 1.2B on T4)
    if cfg.get("lora", False):
        from peft import LoraConfig, TaskType, get_peft_model

        lora_cfg = LoraConfig(
            r=cfg.get("lora_r", 16),
            lora_alpha=cfg.get("lora_alpha", 32),
            lora_dropout=cfg.get("lora_dropout", 0.05),
            task_type=TaskType.CAUSAL_LM,
            target_modules=cfg.get(
                "target_modules",
                [
                    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.out_proj",
                    "feed_forward.w1", "feed_forward.w2", "feed_forward.w3",
                    "conv.in_proj", "conv.out_proj",
                ],
            ),
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    # Gradient checkpointing: trades ~30% speed for ~5-10x activation memory cut.
    # Essential for full-FT 350M fp32 on a 16GB T4.
    if cfg.get("gradient_checkpointing", True):
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        logger.info("Gradient checkpointing enabled")

    logger.info("Loading corpus...")
    max_stories = cfg.get("lcpt_max_stories")
    ds = load_text_corpus(cfg["dataset_path"], max_stories=max_stories, seed=cfg.get("seed", 42))
    ds = chunk_text(ds, tokenizer, max_len, batch_size=cfg.get("tokenize_batch_size", 512))
    logger.info(f"Corpus: {len(ds)} chunks")

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_len,
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
        bf16=use_bf16,
        fp16=use_fp16,
        gradient_checkpointing=cfg.get("gradient_checkpointing", True),
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
        processing_class=tokenizer,
    )

    logger.info("Starting LCPT training...")
    trainer.train()

    out_dir = Path(cfg["output_dir"]) / "lcpt_model"
    if cfg.get("lora", False):
        # Save adapter, then merge into base for downstream SFT
        model.save_pretrained(str(out_dir))
        logger.info(f"✅ LoRA adapter saved to {out_dir}")
        merged = model.merge_and_unload()
        merged_dir = Path(cfg["output_dir"]) / "lcpt_merged"
        merged.save_pretrained(str(merged_dir))
        tokenizer.save_pretrained(str(merged_dir))
        logger.info(f"✅ Merged LCPT model saved to {merged_dir}")
    else:
        trainer.save_model(str(out_dir))
        tokenizer.save_pretrained(str(out_dir))
        logger.info(f"✅ LCPT model saved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/lcpt_config.yaml")
    args = parser.parse_args()
    run_lcpt(args.config)
