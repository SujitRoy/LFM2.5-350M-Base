#!/usr/bin/env python3
"""
Download public Hindi/Hinglish datasets from HuggingFace Hub.
Usage: python3.13 scripts/download_data.py [--output_dir data/raw]
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def write_jsonl_row(f, row: dict):
    """Write one valid JSON row (proper Unicode escaping, no Python repr)."""
    f.write(json.dumps(row, ensure_ascii=False) + "\n")


def download_indic_sft(output_dir: Path):
    """Download Hindi instruction dataset."""
    print("[1/4] Downloading instruct-hindi...")
    ds = load_dataset("mnvkrishna/instruct-hindi", split="train")
    out = output_dir / "instruct_hindi.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for row in ds:
            write_jsonl_row(f, {
                "instruction": row["instruction"],
                "input": row.get("input", "") or "",
                "output": row["output"],
                "language": "hindi",
            })
    print(f"  → {out} ({len(ds)} rows)")
    return len(ds)


def download_indic_conversational(output_dir: Path):
    """Download AI4Bharat conversational dataset (Hindi)."""
    print("[2/4] Downloading AI4Bharat Conversational...")
    ds = load_dataset("ai4bharat/indicconversational", "hi", split="train")
    out = output_dir / "indic_conversational.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for row in ds:
            write_jsonl_row(f, {
                "instruction": row.get("question", ""),
                "input": row.get("context", "") or "",
                "output": row.get("answer", ""),
                "language": "hindi",
            })
    print(f"  → {out} ({len(ds)} rows)")
    return len(ds)


def download_mlqa_hi(output_dir: Path):
    """Download MLQA Hindi QA pairs."""
    print("[3/4] Downloading MLQA Hindi...")
    ds = load_dataset("stanfordcds/mlqa", "mlqa-train.hi", split="train")
    out = output_dir / "mlqa_hi.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for row in ds:
            write_jsonl_row(f, {
                "instruction": row["question"],
                "input": "",
                "output": row["answers"]["text"][0],
                "language": "hindi",
            })
    print(f"  → {out} ({len(ds)} rows)")
    return len(ds)


def download_wiki_hi_summary(output_dir: Path):
    """Download Hindi Wikipedia text for LCPT corpus."""
    print("[4/4] Downloading Hindi Wikipedia...")
    ds = load_dataset("wikimedia/wikipedia", "20231101.hi", split="train")
    out = output_dir / "wikihindi_raw.txt"
    with out.open("w", encoding="utf-8") as f:
        for row in ds:
            text = row.get("text", "").strip()
            if len(text) > 100:
                f.write(text + "\n")
    print(f"  → {out} ({ds.num_rows} rows)")
    return ds.num_rows


def main():
    parser = argparse.ArgumentParser(description="Download Hindi/Hinglish datasets")
    parser.add_argument("--output_dir", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    counts = {}
    for name, fn in [
        ("instruct_hindi", download_indic_sft),
        ("indic_conversational", download_indic_conversational),
        ("mlqa_hi", download_mlqa_hi),
        ("wikihindi", download_wiki_hi_summary),
    ]:
        try:
            counts[name] = fn(args.output_dir)
        except Exception as e:
            print(f"  ⚠ {name} failed: {e}")

    total_sft = sum(v for k, v in counts.items() if k != "wikihindi")
    total_lcpt = counts.get("wikihindi", 0)
    print(f"\nSummary: {total_sft} SFT pairs, ~{total_lcpt} LCPT documents downloaded.")


if __name__ == "__main__":
    main()
