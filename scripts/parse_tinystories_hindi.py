#!/usr/bin/env python3
"""
Parse Meyank/Tiny_Stories_Hindi (EN:/HI: alternating-line parallel corpus, ~2.6GB)
into two training artifacts — streaming, memory-bounded:

  1. LCPT corpus  (data/raw/tinystories_hi_lcpt.txt)
     - HI story text, one story per line, ≤ MAX_LINE_CHARS chars
  2. EN→HI translation SFT pairs (data/raw/tinystories_translate.jsonl)
     - alpaca schema: instruction/input/output, capped by --max_pairs

Usage:
  python3.13 scripts/parse_tinystories_hindi.py \
      --input data/raw/tiny_hi.txt \
      --lcpt_output data/raw/tinystories_hi_lcpt.txt \
      --sft_output data/raw/tinystories_translate.jsonl \
      [--max_pairs 25000] [--lcpt_limit 400000]
"""

import argparse
import json
import random
import time
from pathlib import Path

MAX_LINE_CHARS = 8000          # hard cap per story line (~2-3K tokens)
MIN_STORY_CHARS = 100          # skip fragments
MIN_PAIR_CHARS = 60            # translation pairs must be substantive
MAX_PAIR_CHARS = 1500          # keep translation pairs short-ish for SFT


def iter_pairs(path: Path):
    """Stream (en, hi) pairs. File format: 'EN: <text>' then 'HI: <text>' lines."""
    pending_en = None
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("EN:"):
                pending_en = line[3:].strip()
            elif line.startswith("HI:") and pending_en:
                hi = line[3:].strip()
                if hi:
                    yield pending_en, hi
                pending_en = None


def main():
    parser = argparse.ArgumentParser(description="Parse TinyStories-Hindi parallel corpus")
    parser.add_argument("--input", type=Path, default=Path("data/raw/tiny_hi.txt"))
    parser.add_argument("--lcpt_output", type=Path,
                        default=Path("data/raw/tinystories_hi_lcpt.txt"))
    parser.add_argument("--sft_output", type=Path,
                        default=Path("data/raw/tinystories_translate.jsonl"))
    parser.add_argument("--max_pairs", type=int, default=25000,
                        help="Number of EN->HI translation SFT pairs to emit")
    parser.add_argument("--lcpt_limit", type=int, default=400000,
                        help="Max stories written to LCPT corpus")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    t0 = time.time()

    for out in (args.lcpt_output, args.sft_output):
        out.parent.mkdir(parents=True, exist_ok=True)

    # Reservoir sampling for translation pairs: single pass, uniform sample.
    pair_sample: list[tuple[str, str]] = []
    n_seen = 0
    n_lcpt = 0
    n_pairs_written = 0

    lcpt_f = args.lcpt_output.open("w", encoding="utf-8")

    TRANSLATE_INSTRUCTIONS = [
        "Translate the following story to Hindi:",
        "निम्नलिखित कहानी का हिंदी में अनुवाद करें:",
        "Translate this English text into natural Hindi:",
    ]

    try:
        for en, hi in iter_pairs(args.input):
            n_seen += 1

            # --- LCPT corpus ---
            if n_lcpt < args.lcpt_limit and MIN_STORY_CHARS <= len(hi) <= MAX_LINE_CHARS:
                lcpt_f.write(hi + "\n")
                n_lcpt += 1

            # --- translation SFT reservoir ---
            if MIN_PAIR_CHARS <= len(en) <= MAX_PAIR_CHARS and MIN_PAIR_CHARS <= len(hi) <= MAX_PAIR_CHARS:
                n_pairs_written += 1
                if len(pair_sample) < args.max_pairs:
                    pair_sample.append((en, hi))
                else:
                    j = rng.randrange(n_pairs_written)
                    if j < args.max_pairs:
                        pair_sample[j] = (en, hi)
    finally:
        lcpt_f.close()

    rng.shuffle(pair_sample)
    with args.sft_output.open("w", encoding="utf-8") as f:
        for en, hi in pair_sample:
            row = {
                "instruction": rng.choice(TRANSLATE_INSTRUCTIONS),
                "input": en,
                "output": hi,
                "language": "hindi_translation",
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    elapsed = time.time() - t0
    print(f"✅ Parsed {n_seen:,} EN↔HI pairs in {elapsed:.1f}s")
    print(f"   LCPT corpus : {n_lcpt:,} stories → {args.lcpt_output}")
    print(f"   Translation : {len(pair_sample):,} pairs (of {n_pairs_written:,} eligible) → {args.sft_output}")


if __name__ == "__main__":
    main()
