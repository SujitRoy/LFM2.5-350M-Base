#!/usr/bin/env python3
"""
Convert alpaca-style JSONL (instruction/input/output) to LEAP Finetune
messages format: {"id": ..., "messages": [{"role": ..., "content": ...}]}
Also validates required fields and drops malformed rows.

Usage:
  python3.13 scripts/convert_to_leap_format.py \
      --input data/validated/sft_train.jsonl \
      --output data/leap/sft_train.jsonl
"""

import argparse
import json
import uuid
from pathlib import Path


def _as_str(value) -> str:
    """Coerce a field to str: lists of strings are joined; others ignored."""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list) and value and isinstance(value[0], str):
        return " ".join(v.strip() for v in value if isinstance(v, str)).strip()
    return ""


def convert_row(row: dict, idx: int) -> dict | None:
    instruction = _as_str(row.get("instruction"))
    output = _as_str(row.get("output"))
    inp = _as_str(row.get("input"))

    if not instruction or not output:
        return None

    user_content = f"{instruction}\n\n{inp}" if inp else instruction
    return {
        "id": row.get("id") or f"sample-{idx}-{uuid.uuid4().hex[:8]}",
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": output},
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert alpaca JSONL → LEAP messages format")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    kept, dropped = 0, 0
    with args.input.open(encoding="utf-8") as fin, args.output.open("w", encoding="utf-8") as fout:
        for idx, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                dropped += 1
                continue
            converted = convert_row(row, idx)
            if converted is None:
                dropped += 1
                continue
            fout.write(json.dumps(converted, ensure_ascii=False) + "\n")
            kept += 1

    print(f"✅ Converted {kept} rows → {args.output} (dropped {dropped})")


if __name__ == "__main__":
    main()
