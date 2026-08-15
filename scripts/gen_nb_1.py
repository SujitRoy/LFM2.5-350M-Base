#!/usr/bin/env python3
"""Notebook builder part 1/4: setup cells. Creates cells_state.json."""
import json
from pathlib import Path

HERE = Path(__file__).parent
STATE = HERE / "cells_state.json"

cells = []

def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text})

def code(text):
    cells.append({
        "cell_type": "code", "metadata": {}, "execution_count": None,
        "outputs": [], "source": text,
    })

# ─────────────────────────────────────────────────────────
md("""# 🇮🇳 Fine-tune LFM2.5-350M for Hindi + Hinglish — Complete Beginner Guide

This notebook takes Liquid AI's **LFM2.5-350M-Base** model and teaches it Hindi and
Hinglish (Hindi written in English letters, like "aap kaise ho?") on a **free
Google Colab T4 GPU**.

## What we'll do (overview)
| Step | What | Time on T4 |
|---|---|---|
| 0 | Check GPU & connect Drive | 2 min |
| 1 | Install tools | 5 min |
| 2 | Get the code | 1 min |
| 3 | Download + prepare Hindi/Hinglish data | 10-20 min |
| 4 | **LCPT** — teach the model to read Hindi (raw stories) | 60-90 min |
| 5 | **SFT** — teach it to chat in Hindi/Hinglish | 30-50 min |
| 6 | Test it live | 5 min |
| 7 | (Optional) Export GGUF for Ollama/llama.cpp | 10 min |
| 8 | Save everything to Drive / Hugging Face | 5 min |

## How to use this notebook
- Click the ▶️ button on the left of each cell, **in order, top to bottom**
- Wait for each cell to finish (spinner stops) before running the next
- `!` at the start of a line = a terminal command; lines without it = Python

**Runtime → Change runtime type → T4 GPU** should already be set. Cell 1 checks it.""")

# ─────────────────────────────────────────────────────────
md("""## Step 0 — Check your GPU

Colab gives you a free T4 GPU (16GB). This cell confirms it's active.
If it prints "NO GPU", go to **Runtime → Change runtime type → T4 GPU → Save**, then re-run.""")

code("""# Check GPU
!nvidia-smi

import torch
print(f"\\nPyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("⚠️ NO GPU! Runtime → Change runtime type → T4 GPU → Save, then re-run this cell.")""")

# ─────────────────────────────────────────────────────────
md("""## Step 1 — Mount Google Drive

Your Colab files are **wiped when the session ends**. Google Drive keeps your
trained model safe. This opens a permission popup — click your account → Allow.""")

code("""from google.colab import drive
drive.mount('/content/drive')

# Create our project folder in Drive (where the final model will be saved)
import os
os.makedirs('/content/drive/MyDrive/lfm25_hindi', exist_ok=True)
print("✅ Drive mounted — model will be saved to /content/drive/MyDrive/lfm25_hindi")""")

# ─────────────────────────────────────────────────────────
md("""## Step 2 — Install the tools

- `transformers`, `peft`, `trl` — the finetuning stack (Hugging Face)
- `datasets` — to download training data
- `sentencepiece` — the tokenizer format LFM2.5 uses

⏱️ Takes ~3-5 minutes. Ignore any "pip dependency resolver" warnings — they're harmless.""")

code("""# %%time
%pip install -q -U transformers peft trl datasets accelerate sentencepiece pyyaml
print("✅ Installed")""")

# ─────────────────────────────────────────────────────────
md("""## Step 3 — Get the project code

Clones your GitHub repo (training scripts + configs). Everything we do next
uses these scripts.""")

code("""%cd /content
!git clone https://github.com/SujitRoy/LFM2.5-350M-Base.git
%cd /content/LFM2.5-350M-Base
!ls scripts/ configs/""")

with STATE.open("w") as f:
    json.dump(cells, f, ensure_ascii=False)

print(f"part 1: {len(cells)} cells → {STATE.name}")
