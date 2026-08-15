# LFM2.5-350M Hindi/Hinglish Fine-Tuning

Finetune Liquid AI's **LFM2.5-350M-Base** (hybrid conv+attention, ~350M params) for **Hindi and Hinglish** (code-mixed Hindi-English) support using parameter-efficient fine-tuning.

## Quick Start

### Option A: LEAP Finetune (RECOMMENDED — Liquid AI's official toolkit)
SFT/LoRA + evals + **GGUF export** in one repo. Correct LFM2 module names built in:
```bash
git clone https://github.com/Liquid4All/leap-finetune.git && cd leap-finetune && uv sync
# back in this repo: convert data, then train
python3.13 scripts/convert_to_leap_format.py --input data/validated/sft_train.jsonl --output data/leap/sft_hindi_hinglish_train.jsonl
uv run leap-finetune job_configs/sft_hindi_350m.yaml      # 350M — fits free T4
uv run leap-finetune job_configs/sft_hindi_1.2b.yaml      # 1.2B — needs ~16GB VRAM
uv run leap-finetune export <checkpoint> --quant Q8_0     # GGUF export
```
See `ROADMAP.md` Appendix D for full GGUF quantization guide (Q4_K_M, Q5_K_M, …).

### Option B: Google Colab (free T4 GPU, custom scripts)
1. Open [Colab](https://colab.research.google.com) → New Notebook
2. Copy all cells from `notebooks/colab_notebook.ipynb.tpl` into the notebook
3. Add your HuggingFace token in CELL 1
4. Hit **Runtime → Run all**
5. Total time: ~1–2 hours

### Option C: Local CPU Validation Only
```bash
bash scripts/quickstart.sh
```
Validates environment, downloads data, synthesizes Hinglish samples. Cannot train meaningfully on CPU — use Colab for actual training.

## Project Structure
```
LFM2.5-350M-Base/
├── ROADMAP.md              ← Full technical plan & timeline
├── requirements.txt        ← Python dependencies
├── job_configs/            ← LEAP Finetune job YAMLs (350M, 1.2B, eval)
├── configs/
│   ├── sft_config.yaml     ← LoRA SFT hyperparameters (custom path)
│   └── lcpt_config.yaml    ← Continued pretraining config
├── scripts/
│   ├── download_data.py    ← Fetch Hindi datasets from HF Hub
│   ├── synthesize_hinglish.py  ← Generate Hinglish instruction data
│   ├── validate_data.py    ← Tokenize, dedup, split train/val/test
│   ├── convert_to_leap_format.py ← alpaca JSONL → LEAP messages format
│   ├── run_lcpt.py         ← Continued pretraining trainer
│   ├── run_sft.py          ← LoRA SFT trainer (TRL + PEFT)
│   ├── evaluate.py         ← Hindi/Hinglish/English eval suite
│   ├── demo.py             ← Interactive inference demo
│   └── quickstart.sh       ← One-command setup validation
└── notebooks/
    └── colab_notebook.ipynb.tpl  ← Copy-paste Colab notebook
```

## Model Specs
| | |
|---|---|
| Base | `LiquidAI/LFM2.5-350M-Base` |
| Architecture | 10× LIV double-gated conv + 6× GQA attention (16 layers) |
| Parameters | ~350M (677 MB safetensors) |
| Vocab | 65,536 (SentencePiece, handles Devanagari natively) |
| Context | 128K tokens |
| Training method | LoRA (rank 16, alpha 32) on LFM2 attention + FFN + conv projections |

## Hardware Requirements
| Phase | GPU Required? | Est. Time (T4) |
|---|---|---|
| Data download + synthesis | ❌ CPU OK | 2 min |
| Data validation | ❌ CPU OK | 1 min |
| LCPT (optional) | ✅ Recommended | ~1 hr |
| LoRA SFT | ✅ Required | ~20 min |
| Evaluation | ✅ Recommended | ~5 min |

## Training Pipeline (2 phases)

**Phase 1 — Continued Pretraining (LCPT)** *(optional but recommended)*
- Feed Hindi/Hinglish text through standard causal LM loss
- Warms up the model's Hindi token distributions
- Skip if < 200K LCPT tokens available

**Phase 2 — LoRA SFT**
- 3 epochs, cosine LR schedule, max seq length 1024
- Targets: `self_attn.q/k/v_proj`, `self_attn.out_proj` (⚠️ LFM2 has **no** `o_proj`),
  `feed_forward.w1/w2/w3`, optionally `conv.in_proj/out_proj`
- Data mix: 50% Hindi / 35% Hinglish / 15% English

## Results Validation
After training, `scripts/evaluate.py` checks:
1. **Hindi generation** — Does output contain Devanagari script?
2. **Hinglish code-mixing** — Natural Hindi-English alternation?
3. **English retention** — No catastrophic forgetting on English tasks?

## Known Risks
- CPU training will be prohibitively slow; **use Colab**
- Small model (350M) has limited capacity for complex reasoning
- Hinglish slang may need post-training vocabulary expansion
- Test English degradation carefully after each epoch

## References
- [LFM2.5 Model Card](https://huggingface.co/LiquidAI/LFM2.5-350M-Base)
- [LFM2 Technical Report (arXiv)](https://arxiv.org/abs/2511.23404)
- [Transformers LFM2 Documentation](https://huggingface.co/docs/transformers/v4.57.0/model_doc/lfm2)
