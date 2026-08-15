#!/usr/bin/env python3
"""
Download verified Hindi/Hinglish datasets from HuggingFace Hub.

All datasets below were manually verified (schema + content) on 2026-08-15:
  1. Sujalvc/hinglish-instruct-dataset      10,378 rows — REAL code-mixed Hinglish
  2. iamshnoo/alpaca-cleaned-hindi          51,760 rows — clean Hindi alpaca
  3. ai4bharat/indic-instruct-data-v0.1     7,577 rows  — Hindi multi-turn chat (anudesh/hi)
  4. FreedomIntelligence/evol-instruct-hindi 59,022 rows — Hindi evol-instruct (complex prompts)
  5. atharvanighot/Hindi-Instruct-500K   508,609 rows — Hindi Q&A (question/answer schema)
  6. wikimedia/wikipedia 20231101.hi        ~152K docs  — Hindi Wikipedia for LCPT

Usage: python3.13 scripts/download_data.py [--output_dir data/raw] [--skip_wiki]
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def write_jsonl_row(f, row: dict):
    """Write one valid JSON row (proper Unicode escaping, no Python repr)."""
    f.write(json.dumps(row, ensure_ascii=False) + "\n")


def download_hinglish_instruct(output_dir: Path):
    """Real code-mixed Hinglish instruction data (not romanized Hindi)."""
    print("[1/6] Downloading Sujalvc/hinglish-instruct-dataset...")
    ds = load_dataset("Sujalvc/hinglish-instruct-dataset", split="train")
    out = output_dir / "hinglish_instruct.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for row in ds:
            write_jsonl_row(f, {
                "instruction": row["instruction"],
                "input": row.get("input", "") or "",
                "output": row["output"],
                "language": "hinglish",
            })
    print(f"  → {out} ({len(ds)} rows)")
    return len(ds)


def download_alpaca_hindi(output_dir: Path):
    """Clean Hindi alpaca (instruction/input/output)."""
    print("[2/6] Downloading iamshnoo/alpaca-cleaned-hindi...")
    ds = load_dataset("iamshnoo/alpaca-cleaned-hindi", split="train")
    out = output_dir / "alpaca_hindi.jsonl"
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


def download_indic_instruct_hi(output_dir: Path):
    """AI4Bharat Hindi multi-turn chat (messages format)."""
    print("[3/6] Downloading ai4bharat/indic-instruct-data-v0.1 (anudesh/hi)...")
    ds = load_dataset("ai4bharat/indic-instruct-data-v0.1", "anudesh", split="hi")
    out = output_dir / "indic_anudesh_hi.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for row in ds:
            write_jsonl_row(f, {
                "instruction": row["messages"][0]["content"],
                "input": "",
                "output": row["messages"][1]["content"],
                "language": "hindi",
            })
    print(f"  → {out} ({len(ds)} rows)")
    return len(ds)


def download_evol_hindi(output_dir: Path):
    """Evol-instruct Hindi (complex instruction following)."""
    print("[4/6] Downloading FreedomIntelligence/evol-instruct-hindi...")
    ds = load_dataset("FreedomIntelligence/evol-instruct-hindi", split="train")
    out = output_dir / "evol_hindi.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for row in ds:
            write_jsonl_row(f, {
                "instruction": row["conversations"][0]["value"],
                "input": "",
                "output": row["conversations"][1]["value"],
                "language": "hindi",
            })
    print(f"  → {out} ({len(ds)} rows)")
    return len(ds)


def download_hindi_instruct_500k(output_dir: Path):
    """Large Hindi Q&A dataset (question/answer schema, 508K rows)."""
    print("[5/6] Downloading atharvanighot/Hindi-Instruct-500K...")
    ds = load_dataset("atharvanighot/Hindi-Instruct-500K", split="train")
    out = output_dir / "hindi_instruct_500k.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for row in ds:
            write_jsonl_row(f, {
                "instruction": row["question"],
                "input": "",
                "output": row["answer"],
                "language": "hindi",
            })
    print(f"  → {out} ({len(ds)} rows)")
    return len(ds)


def download_wiki_hi(output_dir: Path):
    """Hindi Wikipedia text for LCPT corpus."""
    print("[6/6] Downloading Hindi Wikipedia (large — takes a while)...")
    ds = load_dataset("wikimedia/wikipedia", "20231101.hi", split="train")
    out = output_dir / "wikihindi_raw.txt"
    with out.open("w", encoding="utf-8") as f:
        for row in ds:
            text = row.get("text", "").strip()
            if len(text) > 100:
                f.write(text + "\n")
    print(f"  → {out} ({ds.num_rows} docs)")
    return ds.num_rows


def main():
    parser = argparse.ArgumentParser(description="Download verified Hindi/Hinglish datasets")
    parser.add_argument("--output_dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--skip_wiki", action="store_true",
                        help="Skip Hindi Wikipedia (LCPT corpus, ~1GB)")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tasks = [
        ("hinglish_instruct", download_hinglish_instruct),
        ("alpaca_hindi", download_alpaca_hindi),
        ("indic_anudesh_hi", download_indic_instruct_hi),
        ("evol_hindi", download_evol_hindi),
        ("hindi_instruct_500k", download_hindi_instruct_500k),
    ]
    if not args.skip_wiki:
        tasks.append(("wikihindi", download_wiki_hi))

    counts = {}
    for name, fn in tasks:
        try:
            counts[name] = fn(args.output_dir)
        except Exception as e:
            print(f"  ⚠ {name} failed: {e}")

    total_sft = sum(v for k, v in counts.items() if k != "wikihindi")
    print(f"\nSummary: {total_sft} SFT pairs, {counts.get('wikihindi', 0)} LCPT docs downloaded.")


if __name__ == "__main__":
    main()
