#!/usr/bin/env python3
"""Notebook generator part 3: SFT training, merge, demo inference."""
import json
from pathlib import Path

HERE = Path(__file__).parent
STATE = HERE / "cells_state.json"
cells = json.loads(STATE.read_text())

def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)})

def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "source": text.splitlines(keepends=True),
                  "execution_count": None, "outputs": []})

md("""\
# Step 8 (Phase 2): Supervised Fine-Tuning (SFT) — teach it to chat

LCPT taught the model to *read* Hindi. SFT now teaches it to *respond*:
it sees formatted conversations (user asks → assistant answers in Hindi/Hinglish)
and learns the response behaviour.

**What happens:**
- LoRA adapters (rank 16, alpha 32) attach to the model — only ~1-2% of weights train
- Loss is computed on the assistant's answer tokens only
- 40,000 curated examples (Hindi + Hinglish + translation pairs), 3 epochs
- fp16 on T4, gradient checkpointing on
- **Expected time on T4: 30-60 minutes**

The config `configs/sft_config_t4.yaml` starts from the LCPT model (`output/lcpt/lcpt_model`).
If you skipped LCPT, first run the next cell to switch it to the base model.
""")

code("""\
# If you SKIPPED the LCPT step (Step 6), run this cell so SFT starts from the base model.
# If you ran LCPT, skip this cell.
import yaml, pathlib
p = pathlib.Path("configs/sft_config_t4.yaml")
cfg = yaml.safe_load(p.read_text())
cfg["model_name_or_path"] = "LiquidAI/LFM2.5-350M-Base"
p.write_text(yaml.dump(cfg, allow_unicode=True, sort_keys=False))
print("sft config now starts from:", cfg["model_name_or_path"])
""")

code("""\
# %%time
# --- SFT TRAINING (30-60 min on T4) ---
!python scripts/run_sft.py --config configs/sft_config_t4.yaml
""")

md("""\
## Step 9 — What just happened? Where are the outputs?

After training finishes you will find:

```
output/sft_lora_t4/
├── adapter/     ← LoRA adapter only (~10-20 MB) — needs the base model to run
├── merged/      ← full standalone model (~680 MB) — this is your final model
└── training/    ← checkpoints + logs
```

The `merged/` folder is a complete HuggingFace model you can share, quantize, or run anywhere.

Let's immediately test it — chat with your Hindi model!
""")

code("""\
# %%time
# --- TEST YOUR FINE-TUNED MODEL ---
!python scripts/demo.py --model_path output/sft_lora_t4/merged
""")

code("""\
# Chat interactively — type your own Hindi / Hinglish prompts
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "output/sft_lora_t4/merged"
tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.float16, trust_remote_code=True
).cuda()
model.eval()

def chat(prompt, max_new_tokens=200):
    text = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True,
    )
    ids = tok(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=max_new_tokens,
                             do_sample=True, temperature=0.7, top_p=0.9)
    return tok.decode(out[0][ids["input_ids"].shape[-1]:], skip_special_tokens=True)

# Try your own prompts here:
for p in [
    "नमस्ते! आप कैसे हैं?",
    "Bhai, mujhe Python seekhna hai, kahan se start karun?",
    "एक छोटी सी कहानी सुनाओ।",
    "What is the capital of India? Answer briefly.",
]:
    print("🧑 USER:", p)
    print("🤖 MODEL:", chat(p))
    print("-" * 60)
""")

STATE.write_text(json.dumps(cells))
print(f"part 3: {len(cells)} cells total")
