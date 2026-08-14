# 🚀 LFM2.5-350M Hindi/Hinglish Fine-Tuning — Google Colab
# Run this notebook end-to-end on Colab (free T4 GPU)

# ─────────────────────────────────────────────────────────
# CELL 1: Install dependencies
# ─────────────────────────────────────────────────────────
!pip install -q transformers==5.15.0 peft==0.20.0 trl==1.10.0 accelerate==1.14.0 datasets bitsandbytes sentencepiece huggingface_hub tensorboard

import torch
print(f"✅ CUDA: {torch.cuda.is_available()} | GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_mem/1e9:.1f}GB")

from huggingface_hub import login
# TODO: Add your HuggingFace token below (Settings → Access Tokens)
# login(token="hf_xxxxxxxxxxxxxxxx")

# ─────────────────────────────────────────────────────────
# CELL 2: Clone repo & setup directories
# ─────────────────────────────────────────────────────────
!git clone https://github.com/YOUR_USERNAME/LFM2.5-350M-Hindi.git /content/LFM2.5-350M-Hindi
%cd /content/LFM2.5-350M-Hindi
!mkdir -p data/{raw,validated} scripts output logs

# ─────────────────────────────────────────────────────────
# CELL 3: Download training data
# ─────────────────────────────────────────────────────────
!python3.13 scripts/download_data.py --output_dir data/raw

# ─────────────────────────────────────────────────────────
# CELL 4: Generate synthetic Hinglish data
# ─────────────────────────────────────────────────────────
!python3.13 scripts/synthesize_hinglish.py --output data/raw/hinglish_synthetic.jsonl --num_samples 300

# ─────────────────────────────────────────────────────────
# CELL 5: Validate & split data
# ─────────────────────────────────────────────────────────
!python3.13 scripts/validate_data.py --output_dir data/validated

# ─────────────────────────────────────────────────────────
# CELL 6: Continued Pretraining (LCPT) — OPTIONAL
# Skip this if you have < 200K LCPT tokens
# ─────────────────────────────────────────────────────────
!python3.13 scripts/run_lcpt.py --config configs/lcpt_config.yaml

# ─────────────────────────────────────────────────────────
# CELL 7: LoRA SFT Training
# ─────────────────────────────────────────────────────────
!python3.13 scripts/run_sft.py --config configs/sft_config.yaml

# ─────────────────────────────────────────────────────────
# CELL 8: Evaluate on test set
# ─────────────────────────────────────────────────────────
!python3.13 scripts/evaluate.py --model_path output/sft_lora/merged --num_tests 5

# ─────────────────────────────────────────────────────────
# CELL 9: Demo inference
# ─────────────────────────────────────────────────────────
!python3.13 scripts/demo.py --base LiquidAI/LFM2.5-350M-Base --adapter output/sft_lora/adapter

# ─────────────────────────────────────────────────────────
# CELL 10: Push to HuggingFace Hub
# ─────────────────────────────────────────────────────────
from huggingface_hub import HfApi, create_repo
api = HfApi()
repo_id = "YOUR_USERNAME/LFM2.5-350M-Hindi"
create_repo(repo_id, repo_type="model", exist_ok=True)
api.upload_folder(folder_path="./output/sft_lora/adapter", repo_id=repo_id, repo_type="model")
api.upload_folder(folder_path="./output/sft_lora/merged", repo_id=repo_id, repo_type="model")
print(f"🎉 Model pushed to https://huggingface.co/{repo_id}")
