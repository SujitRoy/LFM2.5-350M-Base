#!/usr/bin/env python3
"""Notebook generator Part 4: GGUF export + wrap-up. Appends to cells_state.json, writes final .ipynb."""
import json
from pathlib import Path

HERE = Path(__file__).parent
NB_DIR = HERE.parent / "notebooks"
state_path = HERE / "cells_state.json"  # same file parts 1-3 write to
cells = json.loads(state_path.read_text())

md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s}
code = lambda s: {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s}

cells += [
md("""\
## 📦 Step 9 — Export GGUF (run models locally with llama.cpp / Ollama / LM Studio)

GGUF is the file format used by llama.cpp-based tools. We convert the merged model to Q8_0
(near-lossless, ~380MB for 350M). You can change `--quant` to Q4_K_M etc. if you also build
llama.cpp (K-quants need the `llama-quantize` binary from a llama.cpp checkout).

**This step is optional** — skip it if you only want the HF-format model.
"""),
code("""\
# %%time
# Convert merged HF model -> GGUF (F16/BF16/F32/Q8_0 work out of the box)
!cd leap-finetune && uv run leap-finetune export \\
    /content/LFM2.5-350M-Base/output/sft_lora_t4/merged \\
    --quant Q8_0

# List the output
!ls -lh /content/LFM2.5-350M-Base/output/sft_lora_t4/merged/gguf/
"""),
md("""\
## 🧪 Step 10 — Test the GGUF with llama.cpp (optional)

We download a prebuilt llama.cpp binary and run a quick Hindi chat with the quantized model.
If the download link fails, grab a release from https://github.com/ggml-org/llama.cpp/releases
"""),
code("""\
# Download prebuilt llama.cpp (Ubuntu x64 build)
!wget -q https://github.com/ggml-org/llama.cpp/releases/latest/download/llama-bXXXX-bin-ubuntu-x64.zip -O /tmp/llamacpp.zip || echo "download failed - see releases page"
!mkdir -p /tmp/llamacpp && cd /tmp/llamacpp && (unzip -o /tmp/llamacpp.zip 2>/dev/null || echo "no zip")

# Chat test in Hindi (raw prompt format for LFM2.5)
GGUF=$(ls /content/LFM2.5-350M-Base/output/sft_lora_t4/merged/gguf/*.gguf | head -1)
/tmp/llamacpp/llama-cli -m "$GGUF" \\
  -p "<|im_start|>user\\nनमस्ते! आप कैसे हैं?<|im_end|>\\n<|im_start|>assistant\\n" \\
  -n 128 --temp 0.7 2>/dev/null || echo "llama-cli not available — test GGUF locally in Ollama/LM Studio instead"
"""),
md("""\
## 💾 Step 11 — Save everything to Google Drive (IMPORTANT before session ends!)

Colab deletes ALL files when the session ends. Copy the merged model + GGUF to Drive so
you keep your trained model. The merged model (~680MB) + Q8_0 GGUF (~380MB) fit in free Drive.
"""),
code("""\
import shutil, os
from google.colab import drive
drive.mount('/content/drive')

dest = '/content/drive/MyDrive/lfm25-350m-hindi'
os.makedirs(dest, exist_ok=True)

# 1. Merged HF model (the main artifact)
shutil.copytree('/content/LFM2.5-350M-Base/output/sft_lora_t4/merged',
                f'{dest}/merged', dirs_exist_ok=True)

# 2. GGUF quantized model
gguf_dir = '/content/LFM2.5-350M-Base/output/sft_lora_t4/merged/gguf'
if os.path.isdir(gguf_dir):
    shutil.copytree(gguf_dir, f'{dest}/gguf', dirs_exist_ok=True)

# 3. LoRA adapter (small, useful for future merges)
shutil.copytree('/content/LFM2.5-350M-Base/output/sft_lora_t4/adapter',
                f'{dest}/adapter', dirs_exist_ok=True)

print('✅ Saved to Drive:')
for root, dirs, files in os.walk(dest):
    for fn in files:
        p = os.path.join(root, fn)
        print(f'  {os.path.getsize(p)/1e6:8.1f} MB  {p.replace(dest, "")}')
"""),
md("""\
## 🚀 Step 12 — (Optional) Publish to Hugging Face Hub

Share your Hindi/Hinglish model with the world. Create a write token at
https://huggingface.co/settings/tokens and paste it when prompted.
"""),
code("""\
from huggingface_hub import login, HfApi

login()  # paste your HF write token

api = HfApi()
api.create_repo('YOUR_USERNAME/LFM2.5-350M-Hindi-Hinglish', exist_ok=True, repo_type='model')
api.upload_folder(
    folder_path='/content/LFM2.5-350M-Base/output/sft_lora_t4/merged',
    repo_id='YOUR_USERNAME/LFM2.5-350M-Hindi-Hinglish',
    repo_type='model',
)
print('🎉 Model published! Check your HF profile.')
"""),
md("""\
# 🎓 What you just learned (finetuning crash course recap)

| Concept | Where you used it |
|---|---|
| **LCPT / continued pretraining** | Step 7 — raw Hindi stories taught the model Hindi *fluency* (next-word prediction) |
| **SFT / instruction tuning** | Step 8 — conversations taught it to *follow instructions* in Hindi/Hinglish |
| **LoRA** | SFT step — trained tiny adapter matrices instead of all 350M weights (fast, small, no forgetting) |
| **Chat template** | Training data was wrapped in `<|im_start|>user...<|im_end|>` markers the model understands |
| **Merging** | LoRA weights were folded back into the base model for a standalone checkpoint |
| **Quantization / GGUF** | Step 9 — compressed to Q8_0 for llama.cpp/Ollama/LM Studio on your own machine |

## 📈 Ideas for improving the model next
1. **More LCPT**: raise `lcpt_max_stories` to 200000 (3-4h on T4) — biggest quality lever for Hindi
2. **More epochs**: try 4-5 SFT epochs, watch val loss for overfitting
3. **Higher LoRA rank**: r=32, alpha=64 in `configs/sft_config_t4.yaml`
4. **Scale up**: same pipeline with `LiquidAI/LFM2.5-1.2B-Base` (needs `lora: true` for LCPT on T4)
5. **DPO**: after SFT, preference-tune with LEAP's DPO support for nicer response style

## 🆘 Troubleshooting
| Problem | Fix |
|---|---|
| `CUDA out of memory` (SFT) | Halve `per_device_train_batch_size` (4→2), double `gradient_accumulation_steps` |
| Session died mid-training | Reconnect, re-run from Step 1; training resumes are not automatic — checkpoints in `output/*/training/checkpoint-*` can be loaded |
| Very slow download of datasets | It's 2.6GB — normal. Grab a coffee ☕ |
| Loss = nan | Ensure `fp16: true` and `bf16: false` in configs (T4 requirement) |
| Model outputs gibberish English | LCPT was too short — increase `lcpt_max_stories` |
| Hindi quality is meh | It's a 350M model — expect simple, correct Hindi; use 1.2B for better quality |
"""),
]

# ── Assemble final notebook ──
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "colab": {"provenance": [], "gpuType": "T4", "name": "LFM2.5-350M Hindi+Hinglish Finetune"},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
    },
    "cells": cells,
}

out_path = NB_DIR / "LFM2.5-350M_Hindi_Hinglish_T4_Colab.ipynb"
out_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print(f"✅ Final notebook: {out_path}")
print(f"   Total cells: {len(cells)} ({sum(1 for c in cells if c['cell_type']=='code')} code, "
      f"{sum(1 for c in cells if c['cell_type']=='markdown')} markdown)")
print(f"   Size: {out_path.stat().st_size:,} bytes")
