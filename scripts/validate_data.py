#!/usr/bin/env python3
"""
Validate and prepare SFT data for LFM2.5-350M-Hindi training.
Checks tokenization, deduplicates, splits train/val/test.
Usage: python3.13 scripts/validate_data.py --output_dir data/validated
"""

import argparse
import json
import hashlib
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer


MAX_SFT_TOKENS = 1024
MAX_LCPT_LINE_TOKENS = 2048


def load_jsonl(path: Path) -> list[dict]:
    data = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def tokenize_and_validate(text: str, tokenizer, max_tokens: int) -> tuple[int, bool]:
    """Return (token_count, is_valid)."""
    tokens = tokenizer.encode(text, truncation=False, add_special_tokens=False)
    return len(tokens), len(tokens) <= max_tokens


def deduplicate(samples: list[dict], key_fn=None) -> list[dict]:
    """Remove exact duplicates based on instruction+output."""
    seen = set()
    unique = []
    for s in samples:
        if key_fn:
            k = key_fn(s)
        else:
            k = (s.get("instruction", ""), s.get("output", ""))
        h = hashlib.sha256(json.dumps(k, ensure_ascii=False).encode()).hexdigest()[:16]
        if h not in seen:
            seen.add(h)
            unique.append(s)
    return unique


def main():
    parser = argparse.ArgumentParser(description="Validate and prepare SFT data")
    parser.add_argument("--input_dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output_dir", type=Path, default=Path("data/validated"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        "LiquidAI/LFM2.5-350M-Base", trust_remote_code=True
    )

    # --- Load all JSONL sources ---
    all_sft = []
    for fpath in args.input_dir.glob("*.jsonl"):
        if "hinglish" in fpath.name or "instruct" in fpath.name or "mlqa" in fpath.name:
            print(f"Loading {fpath.name}...")
            samples = load_jsonl(fpath)
            all_sft.extend(samples)
            print(f"  → {len(samples)} rows")

    print(f"\nTotal SFT samples before dedup: {len(all_sft)}")

    # --- Deduplicate ---
    all_sft = deduplicate(all_sft)
    print(f"After deduplication: {len(all_sft)}")

    # --- Tokenize & filter ---
    valid, too_long = [], []
    for s in all_sft:
        combined = f"{s.get('instruction', '')} {s.get('input', '')}".strip()
        n_tok, ok = tokenize_and_validate(combined, tokenizer, MAX_SFT_TOKENS)
        if ok:
            s["n_tokens"] = n_tok
            valid.append(s)
        else:
            too_long.append(n_tok)

    print(f"Valid (≤{MAX_SFT_TOKENS} tokens): {len(valid)}")
    print(f"Too long (>1024): {len(too_long)} (avg {sum(too_long)//max(len(too_long),1):d} tokens)")

    # --- Split ---
    import random
    random.seed(42)
    random.shuffle(valid)
    n = len(valid)
    train, val, test = (
        valid[: int(n * 0.8)],
        valid[int(n * 0.8) : int(n * 0.9)],
        valid[int(n * 0.9) :],
    )

    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        out_path = args.output_dir / f"sft_{split_name}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for s in split_data:
                f.write(
                    json.dumps(
                        {
                            "instruction": s["instruction"],
                            "input": s.get("input", ""),
                            "output": s["output"],
                            "language": s.get("language", "unknown"),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        print(f"  {split_name}: {len(split_data)} → {out_path}")

    # --- Build LCPT corpus ---
    lcpt_lines = []
    for fpath in args.input_dir.glob("*.txt"):
        with fpath.open("r", encoding="utf-8") as f:
            lcpt_lines.extend(f.read().strip().split("\n"))
    print(f"\nLCPT raw lines: {len(lcpt_lines)}")

    # Filter and write LCPT
    valid_lcpt = []
    for line in lcpt_lines:
        line = line.strip()
        if len(line) < 10:
            continue
        n_tok, ok = tokenize_and_validate(line, tokenizer, MAX_LCPT_LINE_TOKENS)
        if ok:
            valid_lcpt.append(line)

    lcpt_out = args.output_dir / "lcpt_corpus.txt"
    with lcpt_out.open("w", encoding="utf-8") as f:
        f.write("\n".join(valid_lcpt))
    print(f"LCPT corpus: {len(valid_lcpt)} lines → {lcpt_out}")
    print(f"Total LCPT chars: {sum(len(l) for l in valid_lcpt):,}")

    # --- Validation report ---
    lang_dist = Counter(s.get("language", "unknown") for s in valid)
    report = {
        "total_sft_before_dedup": len(all_sft) + len(too_long),
        "total_sft_after_dedup": len(valid),
        "train": len(train),
        "val": len(val),
        "test": len(test),
        "lcpt_lines": len(valid_lcpt),
        "language_distribution": dict(lang_dist),
        "tokenizer_vocab_size": tokenizer.vocab_size,
    }
    with (args.output_dir / "validation_report.json").open("w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Report saved → {args.output_dir / 'validation_report.json'}")


if __name__ == "__main__":
    main()
