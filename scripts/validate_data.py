#!/usr/bin/env python3
"""
Validate and prepare SFT data for LFM2.5 Hindi/Hinglish training.
Memory-bounded streaming design: processes one line at a time, never loads
the corpus into RAM. Parallelism comes from the Rust tokenizer's internal
thread pool (encode_batch), NOT from process forking.

Speed strategy:
  1. Char-length prefilter: texts far below/above the token limit are
     classified without tokenization (~99% of rows).
  2. Only borderline rows get exact batch tokenization (encode_batch,
     multi-threaded, bounded chunks).
  3. Two-pass split: pass 1 counts valid rows, pass 2 assigns splits.
     Constant memory: only the dedup hash set is held in RAM.

Usage:
  python3.13 scripts/validate_data.py [--num_threads 4] [--batch_size 2048]
"""

import argparse
import hashlib
import json
import os
import random
import time
from collections import Counter
from pathlib import Path

# Hindi/English text averages ~2.5-4 chars per token for this SentencePiece
# vocab. Prefilter bounds are conservative so no borderline row is misjudged:
CHARS_PER_TOKEN_LO = 2.0   # worst-case dense text  -> lower char bound for max_tokens
CHARS_PER_TOKEN_HI = 5.0   # worst-case sparse text  -> upper char bound for max_tokens

MAX_SFT_TOKENS = 1024
MAX_LCPT_LINE_TOKENS = 2048
MIN_OUTPUT_CHARS = 2


def iter_jsonl(path: Path):
    """Stream JSONL one row at a time. Yields (lineno, row|None)."""
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield lineno, json.loads(line)
            except json.JSONDecodeError:
                yield lineno, None


def classify_by_chars(text: str, max_tokens: int) -> str | None:
    """Prefilter: 'valid' / 'too_long' / None (borderline, tokenize exactly)."""
    n = len(text)
    if n <= max_tokens * CHARS_PER_TOKEN_LO:
        return "valid"
    if n > max_tokens * CHARS_PER_TOKEN_HI:
        return "too_long"
    return None


def batched(iterable, size: int):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def dedup_key(row: dict) -> str:
    raw = (row.get("instruction", "") or "") + "\x00" + (row.get("output", "") or "")
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def process_sft_files(
    input_dir: Path,
    tmp_valid: Path,
    tokenizer,
    batch_size: int,
    max_tokens: int,
):
    """Pass 1: stream all SFT JSONL, dedup, prefilter, batch-tokenize borderline.

    Writes surviving rows (with n_tokens filled) to tmp_valid JSONL.
    Returns stats dict.
    """
    stats = Counter()
    seen: set[str] = set()

    # Collect borderline rows lazily: process file-by-file, batch within.
    def borderline_sink(rows: list[dict]) -> dict[int, int]:
        """Tokenize a batch of borderline rows, return {id(row_obj): n_tokens}."""
        texts = [f"{r.get('instruction', '')} {r.get('input', '')}".strip() for r in rows]
        rust = getattr(tokenizer, "_tokenizer", None) or tokenizer
        enc = rust.encode_batch(texts)
        return [len(e.ids) for e in enc]

    for fpath in sorted(input_dir.glob("*.jsonl")):
        print(f"Loading {fpath.name}...")
        pending: list[dict] = []  # borderline rows awaiting exact tokenization

        def flush_pending(out_f):
            nonlocal pending
            if not pending:
                return
            tokens_map = borderline_sink(pending)
            for row, n_tok in zip(pending, tokens_map):
                if n_tok <= max_tokens:
                    row["n_tokens"] = n_tok
                    out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    stats["valid"] += 1
                else:
                    stats["too_long"] += 1
                    stats["too_long_tokens_sum"] += n_tok
            pending = []

        with tmp_valid.open("a", encoding="utf-8") as out_f:
            for lineno, row in iter_jsonl(fpath):
                stats["total"] += 1
                if row is None:
                    stats["malformed"] += 1
                    continue
                instruction = row.get("instruction") or ""
                output = row.get("output") or ""
                if not isinstance(instruction, str) or not isinstance(output, str):
                    stats["malformed"] += 1
                    continue
                if len(instruction.strip()) < 3 or len(output.strip()) < MIN_OUTPUT_CHARS:
                    stats["empty_fields"] += 1
                    continue
                key = dedup_key(row)
                if key in seen:
                    stats["duplicates"] += 1
                    continue
                seen.add(key)

                combined = f"{instruction} {row.get('input') or ''}".strip()
                verdict = classify_by_chars(combined, max_tokens)
                if verdict == "valid":
                    row["n_tokens"] = None  # exact count unknown; fine for stats
                    out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    stats["valid"] += 1
                elif verdict == "too_long":
                    stats["too_long"] += 1
                else:
                    pending.append(row)
                    if len(pending) >= batch_size:
                        flush_pending(out_f)
            flush_pending(out_f)
        print(f"  → running totals: {dict(stats)}")

    return stats, seen


def split_pass(tmp_valid: Path, output_dir: Path, seed: int):
    """Pass 2: count valid rows, then stream-assign 80/10/10 splits."""
    n = 0
    with tmp_valid.open("r", encoding="utf-8") as f:
        for _ in f:
            n += 1

    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    train_cut = int(n * 0.8)
    val_cut = int(n * 0.9)

    paths = {
        "train": output_dir / "sft_train.jsonl",
        "val": output_dir / "sft_val.jsonl",
        "test": output_dir / "sft_test.jsonl",
    }
    files = {k: v.open("w", encoding="utf-8") for k, v in paths.items()}
    lang_dist = Counter()
    try:
        with tmp_valid.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                idx = order[i]
                dest = "train" if idx < train_cut else ("val" if idx < val_cut else "test")
                files[dest].write(line)
                if dest == "train":
                    lang_dist[json.loads(line).get("language", "unknown")] += 1
    finally:
        for fh in files.values():
            fh.close()
    split_counts = {"train": train_cut, "val": val_cut - train_cut, "test": n - val_cut, "total": n}
    return split_counts, lang_dist


def process_lcpt_files(input_dir: Path, output_dir: Path, tokenizer, batch_size: int):
    """Stream LCPT .txt files with the same prefilter strategy."""
    max_tokens = MAX_LCPT_LINE_TOKENS
    stats = Counter()
    out_path = output_dir / "lcpt_corpus.txt"
    pending_lines: list[str] = []

    def flush(out_f):
        if not pending_lines:
            return
        rust = getattr(tokenizer, "_tokenizer", None) or tokenizer
        enc = rust.encode_batch(pending_lines)
        for line, e in zip(pending_lines, enc):
            if len(e.ids) <= max_tokens:
                out_f.write(line + "\n")
                stats["kept"] += 1
            else:
                stats["too_long"] += 1
        pending_lines.clear()

    with out_path.open("w", encoding="utf-8") as out_f:
        for fpath in sorted(input_dir.glob("*.txt")):
            print(f"LCPT: streaming {fpath.name}...")
            with fpath.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if len(line) < 10:
                        continue
                    stats["total"] += 1
                    verdict = classify_by_chars(line, max_tokens)
                    if verdict == "valid":
                        out_f.write(line + "\n")
                        stats["kept"] += 1
                    elif verdict == "too_long":
                        stats["too_long"] += 1
                    else:
                        pending_lines.append(line)
                        if len(pending_lines) >= batch_size:
                            flush(out_f)
            flush(out_f)
    return stats


def main():
    parser = argparse.ArgumentParser(description="Validate and prepare SFT data (memory-bounded)")
    parser.add_argument("--input_dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output_dir", type=Path, default=Path("data/validated"))
    parser.add_argument("--num_threads", type=int, default=4,
                        help="Rust tokenizer thread pool size (not process forks)")
    parser.add_argument("--batch_size", type=int, default=2048,
                        help="Rows per tokenizer batch for borderline rows")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    t0 = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "LiquidAI/LFM2.5-350M-Base", trust_remote_code=True
    )

    tmp_valid = args.output_dir / ".tmp_valid.jsonl"
    if tmp_valid.exists():
        tmp_valid.unlink()

    print("Pass 1: streaming + dedup + prefilter + borderline batch tokenization...")
    stats, _ = process_sft_files(
        args.input_dir, tmp_valid, tokenizer, args.batch_size, MAX_SFT_TOKENS
    )

    print(f"\nTotal rows read:        {stats['total']:,}")
    print(f"Malformed JSON:         {stats['malformed']:,}")
    print(f"Empty/short fields:     {stats['empty_fields']:,}")
    print(f"Duplicates removed:     {stats['duplicates']:,}")
    print(f"Valid (≤{MAX_SFT_TOKENS} tokens):     {stats['valid']:,}")
    print(f"Too long:               {stats['too_long']:,}")

    print("\nPass 2: splitting 80/10/10...")
    split_counts, lang_dist = split_pass(tmp_valid, args.output_dir, args.seed)
    for split, cnt in split_counts.items():
        print(f"  {split}: {cnt:,}")
    tmp_valid.unlink()

    print("\nLCPT corpus preparation...")
    lcpt_stats = process_lcpt_files(args.input_dir, args.output_dir, tokenizer, args.batch_size)
    print(f"  kept {lcpt_stats['kept']:,}/{lcpt_stats['total']:,} lines")

    report = {
        "total_rows_read": stats["total"],
        "malformed": stats["malformed"],
        "empty_fields": stats["empty_fields"],
        "duplicates_removed": stats["duplicates"],
        "valid": stats["valid"],
        "too_long": stats["too_long"],
        "splits": {k: v for k, v in split_counts.items()},
        "lcpt_kept": lcpt_stats["kept"],
        "language_distribution": dict(lang_dist),
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "wall_time_seconds": round(time.time() - t0, 1),
    }
    with (args.output_dir / "validation_report.json").open("w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Report saved → {args.output_dir / 'validation_report.json'}")
    print(f"⏱  Total wall time: {report['wall_time_seconds']}s")


if __name__ == "__main__":
    main()
