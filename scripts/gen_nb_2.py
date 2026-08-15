#!/usr/bin/env python3
"""Notebook generator part 2/4: data download + validation + LCPT."""
import json
from pathlib import Path

HERE = Path(__file__).parent
STATE = HERE / "cells_state.json"

cells = json.loads(STATE.read_text())

def md(text): cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)})
def code(text): cells.append({"cell_type": "code", "metadata": {}, "source": text.splitlines(keepends=True), "execution_count": None, "outputs": []})


md("""\
## Step 4 — Download the datasets (~4 min)

This runs our verified dataset stack (all were manually checked on HF Hub).
Total: **~660,000 Hindi/Hinglish instruction pairs**.
""")

code("""\
# %%time
%cd /content/LFM2.5-350M-Base

# Downloads 5 SFT datasets (~637K pairs). Wikipedia is skipped (we use TinyStories instead)
!python scripts/download_data.py --skip_wiki""")

md("""\
## Step 5 — Download & parse TinyStories-Hindi (the fluency corpus)

A 2.6 GB parallel corpus (484K English↔Hindi story pairs, machine-translated with
IndicTrans2). From it we extract:
- **LCPT corpus**: 400,000 Hindi stories (raw reading practice for the model)
- **25,000 EN→HI translation pairs** (added to SFT mix)
""")

code("""\
# %%time
# ~4 min download
!wget -q -O /content/tiny_hi.txt "https://huggingface.co/datasets/Meyank/Tiny_Stories_Hindi/resolve/main/translations_indictrans2-200m-2.txt"

# Parse into LCPT corpus + translation SFT pairs (~20 s, streams — low RAM)
!python scripts/parse_tinystories_hindi.py \\
    --input /content/tiny_hi.txt \\
    --lcpt_output data/raw/tinystories_hi_lcpt.txt \\
    --sft_output data/raw/tinystories_translate.jsonl \\
    --max_pairs 25000

print("\\n✅ outputs:")
!wc -l data/raw/tinystories_hi_lcpt.txt data/raw/tinystories_translate.jsonl""")

md("""\
## Step 6 — Validate, dedup, token-check, split (~4 min)

Streams every JSONL row: drops malformed/duplicate/over-length rows, checks token
counts with the actual model tokenizer, writes an 80/10/10 train/val/test split,
and builds the LEAP-format training file.
""")

code("""\
# %%time
!python scripts/validate_data.py --num_threads 2 --batch_size 2048

# Convert to LEAP messages format ({"id", "messages": [...]}) — needed for leap-finetune later
!python scripts/convert_to_leap_format.py \\
    --input data/validated/sft_train.jsonl \\
    --output data/leap/sft_hindi_hinglish_train.jsonl
!python scripts/convert_to_leap_format.py \\
    --input data/validated/sft_val.jsonl \\
    --output data/leap/sft_val.jsonl""")

md("""\
# Step 7 (Phase 1): Continued Pre-Training — LCPT

**Goal:** teach the model to *read* Hindi fluently before teaching it to *chat*.

**Why this matters (the key insight of this project):** the LFM2.5 tokenizer was not
optimized for Devanagari — common Hindi words fragment into 4–17 tokens (English
averages ~4 chars/token; Hindi here ~0.6). The base model has seen almost no Hindi,
so SFT alone would produce broken grammar. LCPT on 30,000 natural Hindi stories
builds the missing fluency.

**Config:** `configs/lcpt_config_t4.yaml` → fp16 (T4 has no bf16), 1 epoch,
30K stories (~90–120 min on T4). Loss should drop from ~4 to under 2.5.
""")

code("""\
# %%time
# Sanity-check the config first
!cat configs/lcpt_config_t4.yaml""")

code("""\
# %%time
# ~1.5–2h on T4 (30K stories ≈ 55M tokens). Watch the loss fall — that's the model learning Hindi.
!python scripts/run_lcpt.py --config configs/lcpt_config_t4.yaml""")

md("""\
### ✅ LCPT checkpoint quiz (read before continuing)
- The loss went from ~4 → ~2 (if much higher, something's wrong — check the logs)
- `output/lcpt/lcpt_model/` now exists and contains `model.safetensors`
- If Colab disconnected during training: just re-run the cells above — data survives
  (it's on disk), but training restarts from scratch. For long runs consider saving
  checkpoints to Drive.
""")

STATE.write_text(json.dumps(cells, ensure_ascii=False))
print(f"part 2: {len(cells)} cells total")
