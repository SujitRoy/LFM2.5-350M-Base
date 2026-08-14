#!/usr/bin/env python3
"""
Download public Hindi/Hinglish datasets from HuggingFace Hub.
Usage: python3.13 scripts/download_data.py [--output_dir data/raw]
"""

import argparse
import os
from pathlib import Path

from datasets import load_dataset


def download_indic_sft(output_dir: Path):
    """Download AI4Bharat IndicSFT (Hindi subset)."""
    print("[1/4] Downloading AI4Bharat IndicSFT...")
    ds = load_dataset("mnvkrishna/instruct-hindi", split="train")
    out = output_dir / "instruct_hindi.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for row in ds:
            f.write(
                f'{{"instruction": {row["instruction"]!r}, '
                f'"input": {row.get("input", "")!r}, '
                f'"output": {row["output"]!r}}}\n'
            )
    print(f"  → {out} ({len(ds)} rows)")
    return len(ds)


def download_indic_conversational(output_dir: Path):
    """Download AI4Bharat conversational dataset."""
    print("[2/4] Downloading AI4Bharat Conversational...")
    ds = load_dataset("ai4bharat/indicconversational", "hi", split="train")
    out = output_dir / "indic_conversational.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for row in ds:
            f.write(
                f'{{"instruction": {row.get("question", "")!r}, '
                f'"input": {row.get("context", "")!r}, '
                f'"output": {row.get("answer", "")!r}}}\n'
            )
    print(f"  → {out} ({len(ds)} rows)")
    return len(ds)


def download_mlqa_hi(output_dir: Path):
    """Download MLQA Hindi QA pairs."""
    print("[3/4] Downloading MLQA Hindi...")
    ds = load_dataset("stanfordcds/mlqa", "mlqa-train.hi", split="train")
    out = output_dir / "mlqa_hi.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for row in ds:
            f.write(
                f'{{"instruction": {row["question"]!r}, '
                f'"input": "", '
                f'"output": {row["answers"]["text"][0]!r}}}\n'
            )
    print(f"  → {out} ({len(ds)} rows)")
    return len(ds)


def download_wiki_hi_summary(output_dir: Path):
    """Download WikiHindi for LCPT corpus."""
    print("[4/4] Downloading WikiHindi summaries...")
    ds = load_dataset("csebuetnlp/wikihindi", split="train")
    out = output_dir / "wikihindi_raw.txt"
    with out.open("w", encoding="utf-8") as f:
        for row in ds:
            text = row.get("text", "").strip()
            if len(text) > 20:
                f.write(text + "\n")
    print(f"  → {out} ({ds.num_rows} rows)")
    return ds.num_rows


def main():
    parser = argparse.ArgumentParser(description="Download Hindi/Hinglish datasets")
    parser.add_argument("--output_dir", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    counts = {}
    try:
        counts["instruct_hindi"] = download_indic_sft(args.output_dir)
    except Exception as e:
        print(f"  ⚠ instruct-hindi failed: {e}")

    try:
        counts["indic_conversational"] = download_indic_conversational(args.output_dir)
    except Exception as e:
        print(f"  ⚠ indicconversational failed: {e}")

    try:
        counts["mlqa_hi"] = download_mlqa_hi(args.output_dir)
    except Exception as e:
        print(f"  ⚠ mlqa failed: {e}")

    try:
        counts["wikihindi"] = download_wiki_hi_summary(args.output_dir)
    except Exception as e:
        print(f"  ⚠ wikihindi failed: {e}")

    total_sft = sum(v for k, v in counts.items() if k != "wikihindi")
    total_lcpt = counts.get("wikihindi", 0)
    print(f"\nSummary: {total_sft} SFT pairs, ~{total_lcpt} LCPT sentences downloaded.")


if __name__ == "__main__":
    main()
