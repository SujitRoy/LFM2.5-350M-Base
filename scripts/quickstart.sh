#!/bin/bash
# Quick-start shell script for local development (CPU validation only)
# For actual training, use the Colab notebook.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "=== LFM2.5-350M Hindi/Hinglish Finetune — Setup & Validation ==="
echo ""

# 0. Check environment
echo "[0] Checking environment..."
python3.13 -c "
import torch, transformers, peft, trl, datasets, accelerate
print(f'  torch={torch.__version__}  transformers={transformers.__version__}')
print(f'  peft={peft.__version__}  trl={trl.__version__}')
print(f'  CUDA: {torch.cuda.is_available()}')
" || { echo "❌ Missing packages. Run: pip3.13 install -r requirements.txt"; exit 1; }

# 1. Download base model (local copy)
echo ""
echo "[1] Downloading base model..."
huggingface-cli download LiquidAI/LFM2.5-350M-Base --local-dir models/base 2>/dev/null || echo "  ⚠ Model already cached or HF access restricted"

# 2. Download data
echo ""
echo "[2] Downloading training data..."
python3.13 scripts/download_data.py --output_dir data/raw

# 3. Synthesize Hinglish
echo ""
echo "[3] Generating Hinglish synthetic data..."
python3.13 scripts/synthesize_hinglish.py --output data/raw/hinglish_synthetic.jsonl --num_samples 200

# 4. Validate data
echo ""
echo "[4] Validating data..."
python3.13 scripts/validate_data.py --output_dir data/validated

# 5. Show data summary
echo ""
echo "[5] Data summary:"
cat data/validated/validation_report.json 2>/dev/null || echo "  (run validation first)"

# 6. Quick inference test on base model
echo ""
echo "[6] Quick base model test (CPU):"
python3.13 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
tok = AutoTokenizer.from_pretrained('models/base', trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained('models/base', torch_dtype=torch.float32)
inp = tok('नमस्तே', return_tensors='pt')
with torch.no_grad():
    out = model.generate(**inp, max_new_tokens=20, do_sample=False)
print('  Base model output:', tok.decode(out[0], skip_special_tokens=True))
" 2>/dev/null || echo "  ⚠ Inference test skipped (model may need download)"

echo ""
echo "✅ Setup complete. Next step: run Colab notebook for GPU training."
echo "   See notebooks/colab_notebook.ipynb.tpl for full training pipeline."
