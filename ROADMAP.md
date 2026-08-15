# Finetuning Roadmap: LFM2.5-350M-Base → Hindi/Hinglish Support

## Executive Summary

| Item | Detail |
|---|---|
| **Base Model** | `LiquidAI/LFM2.5-350M-Base` (350M params, hybrid conv+attention) |
| **Target Capability** | Hindi + Hinglish instruction following & conversation |
| **Hardware** | CPU-only server → **train on Google Colab (free T4)** or Kaggle |
| **Method** | 2-phase: Continued Pretraining (LCPT) → LoRA SFT |
| **Tokenizer** | Reused as-is — SentencePiece already handles Devanagari reasonably |
| **Total Estimated GPU Hours** | ~4h on T4 (1h LCPT + 3h SFT) |
| **Estimated Data Size** | 500K–1M tokens LCPT + 8K–15K pairs SFT |
| **Expected Output** | `LiquidAI/LFM2.5-350M-Hindi` on Hugging Face |

---

## Phase 0: Environment Setup (Day 1)

### 0.1 Server Readiness Check
```bash
# Verify all packages are available
python3.13 -c "import transformers, torch, peft, trl, datasets, accelerate"
# Result: all OK on this machine
```

### 0.2 Create Reproducible Environment File
→ See `requirements.txt`

### 0.3 Hugging Face Hub Account
- Create account at huggingface.co
- Generate access token: Settings → Access Tokens → New token (read + write)
- Set env var: `export HF_TOKEN=hf_xxx`

---

## Phase 1: Data Collection & Preparation (Days 1–3)

### 1.1 Public Data Sources (download scripts in `scripts/download_data.py`)

| Dataset | Source | Format | Size | License |
|---|---|---|---|---|
| IndicBERT-instruct-hi | HuggingFace `mnvkrishna/instruct-hindi` | JSONL | ~10K pairs | MIT |
| AI4Bharat Conversational | HuggingFace `ai4bharat/indicconversational` | Parquet | ~50K turns | Apache 2.0 |
| WikiHindi Summaries | Wikipedia dumps → clean text | Raw text | ~200K sentences | CC-BY-SA |
| BanglaPedia Hindi | Scraped from Banglapedia | Raw text | ~50K articles | Free |
| Hinglish Conversations | Synthetic via LLM API | JSONL | 10–20K pairs | — |
| Hindi SQuAD / MLQA | HuggingFace `squad_v2` filtered for hi | JSON | ~1K QA pairs | — |
| IndicTrans2 Parallel | IIT Bombay parallel corpus | Parallel | ~10M sentence pairs | Research |

### 1.2 Synthetic Hinglish Generation (script: `scripts/synthesize_hinglish.py`)
```python
# Use GPT-4o-mini or Claude Haiku to translate/generate
# Prompt template converts English instructions to natural Hinglish
# Target: 15,000 diverse examples covering:
#   - Casual chat (30%)
#   - Q&A / knowledge (25%)
#   - Coding help (20%)
#   - Translation requests (15%)
#   - Creative writing / storytelling (10%)
```

### 1.3 Data Validation Pipeline (`scripts/validate_data.py`)
- Encode each example with tokenizer, check token count
- Remove any example with > 2048 tokens (LCPT) or > 1024 tokens (SFT)
- Deduplicate using MinHash or simple SHA-256 of normalized text
- Split: 80% train / 10% val / 10% test
- Output: `data/validated/{lcpt_text, sft_train.jsonl, sft_val.jsonl, sft_test.jsonl}`

### 1.4 Expected Output
```
data/
├── lcpt_corpus.txt              # 500K-1M tokens, one sentence per line
├── sft_train.jsonl              # {"instruction":..., "input":..., "output":...}
├── sft_val.jsonl
├── sft_test.jsonl
└── validation_report.json
```

---

## Phase 2: Continued Pretraining (LCPT) (Day 4 — Colab)

### 2.1 Objective
Improve the model's exposure to Hindi/Hinglish token distributions before SFT.
This is critical because the base model was trained mostly on English.

### 2.2 Method: Masked Language Modeling (MLM) Lite
Since LFM2 is a causal LM, we use standard next-token prediction on Hindi text
with a longer learning rate warmup.

### 2.3 Training Configuration
```python
training_args = TrainingArguments(
    output_dir="./output/lcpt",
    num_train_epochs=2,
    per_device_train_batch_size=8,        # T4: 8GB VRAM sufficient
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    weight_decay=0.01,
    max_seq_length=1024,
    fp16=True,                            # mixed precision on T4
    logging_steps=50,
    save_strategy="epoch",
    report_to="tensorboard",
)
```

### 2.4 Expected Runtime (T4)
- 500K tokens / (8×4) ≈ 15,625 steps per epoch × 2 epochs ≈ 31K steps
- ~1 step/sec on T4 → **~9 hours wall time**
- Optimization: reduce to 200K tokens for first pass (~3.5 hours)

### 2.5 Evaluation Checkpoints
After each epoch, run quick benchmark:
```python
eval_prompts = [
    ("Translate to English: नमस्ते दुनिया", "Hello world"),
    ("Translate to Hindi: How are you?", "आप कैसे हैं?"),
    ("What is 2+2 in Hindi?", "चार"),
    ("Write a short story in Hindi about a boy and his dog.", "..."),
]
```

### 2.6 Decision Gate
If LCPT benchmarks show >5% improvement over base → proceed to Phase 3.
If not, skip LCPT and go straight to SFT (acceptable for small data).

---

## Phase 3: LoRA Supervised Fine-Tuning (SFT) (Days 5–7 — Colab)

### 3.1 Method: TRL SFTTrainer + PEFT LoRA

```python
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig

# Target ONLY attention projections (preserves conv layer weights)
lora_config = LoraConfig(
    r=16,                           # rank
    lora_alpha=32,                  # scale
    lora_dropout=0.05,
    task_type=TaskType.CAUSAL_LM,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj"  # attention layers only
        # Do NOT target conv layers — they handle edge inference speed
    ],
)
```

### 3.2 Full Training Configuration
```python
sft_config = SFTConfig(
    output_dir="./output/sft_lora",
    dataset_text_field="text",
    max_seq_length=1024,
    packing=False,                    # keep sequences separate for chat format
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    weight_decay=0.01,
    bf16=True,                        # use bf16 if available, else fp16
    logging_steps=20,
    save_strategy="steps",
    save_total_limit=3,
    eval_strategy="epoch",
    report_to="tensorboard",
    seed=42,
)
```

### 3.3 Chat Template Formatting
```python
def format_chat(examples):
    """Convert SFT JSONL to chat messages the model expects."""
    texts = []
    for inst, inp, out in zip(
        examples["instruction"], examples["input"], examples["output"]
    ):
        msg = [
            {"role": "user", "content": inst + (" " + inp if inp else "")},
            {"role": "assistant", "content": out},
        ]
        texts.append(tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=False))
    return {"text": texts}
```

### 3.4 Data Mix Strategy
```
Training set composition:
  50% pure Hindi instructions (from IndicBERT dataset)
  35% Hinglish code-mixed (synthetic + curated)
  15% English (to prevent catastrophic forgetting of base capability)
```

### 3.5 Expected Runtime (T4)
- 10K SFT pairs, seq_len=1024, batch=4×8=32 effective
- ~313 steps/epoch × 3 epochs = ~940 steps
- ~0.8 steps/sec on T4 → **~20 minutes wall time**

### 3.6 Hyperparameter Grid (run on val set)
| Run | lr | r | alpha | epochs | data_mix |
|---|---|---|---|---|---|
| A (default) | 2e-4 | 16 | 32 | 3 | 50/35/15 |
| B | 1e-4 | 8 | 16 | 3 | 50/35/15 |
| C | 2e-4 | 16 | 32 | 3 | 60/30/10 |
| D | 2e-4 | 16 | 32 | 5 | 50/35/15 |

Select best by val loss + manual Hindi quality check.

---

## Phase 4: Merge & Export (Day 7)

### 4.1 Merge LoRA Weights
```python
from peft import PeftModel
base_model = AutoModelForCausalLM.from_pretrained("LiquidAI/LFM2.5-350M-Base")
model = PeftModel.from_pretrained(base_model, "./output/sft_lora/checkpoint-best")
merged = model.merge_and_unload()
merged.save_pretrained("./output/lfm25-350m-hindi-merged")
tokenizer.save_pretrained("./output/lfm25-350m-hindi-merged")
```

### 4.2 Push to Hugging Face Hub
```python
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path="./output/lfm25-350m-hindi-merged",
    repo_id="your-username/LFM2.5-350M-Hindi",
    repo_type="model",
)
```

### 4.3 Output Artifacts
```
output/
├── lcpt/                          # Continued pretraining checkpoints (optional)
├── sft_lora/
│   ├── adapter_config.json        # LoRA config
│   ├── adapter_model.safetensors  # ~5-10 MB adapter weights
│   └── checkpoint-best/           # Best checkpoint by val loss
└── lfm25-350m-hindi-merged/       # Final merged model (~677 MB)
    ├── config.json
    ├── model.safetensors
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── special_tokens_map.json
```

---

## Phase 5: Evaluation (Day 8)

### 5.1 Automated Benchmarks

| Benchmark | Description | Tool |
|---|---|---|
| **Hindi MT** | English→Hindi translation quality | `datasets.load_dataset("wmt21", "en-hi")` |
| **XLORASUM** | Hindi summarization | Custom prompt + BLEU/Rouge |
| **InstructEval-Hi** | Instruction following on Hindi prompts | Custom 50-question suite |
| **MMLU (filtered)** | Multilingual knowledge retention | Check English scores unchanged |
| **Human Eval** | Blind comparison: base vs fine-tuned | 20 Hinglish prompts, rated 1-5 |

### 5.2 Anti-Forgetting Checks
Run these on the **test split** (held-out English prompts):
```python
english_checks = [
    ("Explain quantum entanglement simply.", None),
    ("Write a Python function to merge two sorted lists.", None),
    ("What is the capital of France?", None),
]
# Compare BLEU/ROUGE against base model — must not degrade >2%
```

### 5.3 Failure Mode Analysis
Document these specific risks and how to mitigate:
- **English degradation**: Increase English ratio in data mix, add regularization
- **Hindi script confusion** (अ vs आ): Add more Devanagari examples
- **Code-switching instability**: Ensure Hinglish examples cover technical domains
- **Repetition loops**: Temperature=0.7, top_p=0.9 during generation

---

## Phase 6: Deployment & Monitoring (Week 2+)

### 6.1 Inference Script (`scripts/serve.py`)
```python
# Single-file server using vLLM or HuggingFace Transformers pipeline
# Supports both Hindi and Hinglish inputs
# Exposes REST endpoint: POST /generate {prompt, max_new_tokens=512}
```

### 6.2 Monitoring Metrics
- Per-language response quality (tracked via daily prompt suite)
- Token latency on target edge device (Raspberry Pi 5 / Snapdragon)
- Hallucination rate on Hindi factual queries

---

## Appendix A: Google Colab One-Liner Setup

```python
# colab_setup.py — paste at top of notebook
!pip install -q transformers==5.15.0 peft==0.20.0 trl==1.10.0 accelerate==1.14.0 datasets bitsandbytes
!pip install -q sentencepiece

import torch
print(f"CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}")
# Should print: CUDA: True, GPU: NVIDIA A100-SXM4-40GB (Colab free tier)
```

---

## Appendix B: Risk Register

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| No GPU access | Medium | High | Colab free tier available; fallback to Kaggle |
| Hindi tokenization gaps | Low | Medium | Test tokenizer coverage first; expand vocab only if needed |
| Catastrophic forgetting | Medium | High | Keep 15% English in SFT; monitor MMLU scores |
| LCPT too slow on small data | High | Low | Skip LCPT if <200K tokens available; go straight to SFT |
| LoRA rank too low for Hindi | Medium | Medium | Start r=16; increase to r=32 if val loss plateaus early |
| Vocab mismatch on Hinglish slang | Medium | Low | Add OOV words to tokenizer if needed (post-initial-SFT) |

---

## Appendix C: Quick Reference Commands

```bash
# Download base model
huggingface-cli download LiquidAI/LFM2.5-350M-Base --local-dir ./models/base

# Quick inference test
python3.13 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained('./models/base', torch_dtype=torch.bfloat16)
tok = AutoTokenizer.from_pretrained('./models/base')
inp = tok('हिंदी में बोलो: नमस्ते', return_tensors='pt')
out = model.generate(**inp, max_new_tokens=20)
print(tok.decode(out[0], skip_special_tokens=True))
"

# Run SFT training (on Colab)
python3.13 scripts/run_sft.py --config configs/sft_config.yaml
```

---

## Appendix D: LEAP Finetune (Liquid's official toolkit) — RECOMMENDED

[leap-finetune](https://github.com/Liquid4All/leap-finetune) is Liquid AI's repo for the full
customization loop: data prep → SFT/DPO/GRPO (LoRA or full FT) → evals → **GGUF export**.
It uses the correct LFM2 module names and supports `Lfm2ForCausalLM` GGUF conversion natively.

### D.1 Setup
```bash
git clone https://github.com/Liquid4All/leap-finetune.git
cd leap-finetune && uv sync        # or: uv sync --no-group cuda --group rocm
```

### D.2 Prepare data (messages format)
```bash
# alpaca-style {instruction, input, output} → {id, messages: [{role, content}]}
python3.13 scripts/convert_to_leap_format.py \
    --input data/validated/sft_train.jsonl \
    --output data/leap/sft_hindi_hinglish_train.jsonl
```

### D.3 Train (job configs in `job_configs/`)
```bash
# 350M: fits any 8GB+ GPU (T4 free tier)
uv run leap-finetune job_configs/sft_hindi_350m.yaml

# 1.2B: needs ~16GB VRAM for bf16 LoRA
uv run leap-finetune job_configs/sft_hindi_1.2b.yaml

# Standalone eval of any checkpoint
uv run leap-finetune eval job_configs/eval_hindi_standalone.yaml --output results.json
```

### D.4 Export GGUF quantized models
```bash
# Direct quants (no llama.cpp build needed): F16 BF16 F32 Q8_0
uv run leap-finetune export <checkpoint_dir> --quant Q8_0

# K-quants (need llama.cpp's llama-quantize binary):
#   build llama.cpp once, then:
uv run leap-finetune export <checkpoint_dir> \
    --quant Q4_K_M --quant Q5_K_M \
    --llama-cpp-dir /path/to/llama.cpp

# LoRA adapter → GGUF (usable as --lora in llama.cpp):
uv run leap-finetune export <adapter_dir> --base-model-path <base_model_dir>
```

Quant selection guide (350M/1.2B on-device):
| Quant | Size (1.2B) | Use when |
|---|---|---|
| Q8_0 | ~1.3 GB | Near-lossless, RAM available |
| Q6_K | ~1.1 GB | Good quality/space tradeoff |
| Q5_K_M | ~0.9 GB | Balanced default for phones |
| Q4_K_M | ~0.8 GB | Max compression, small quality loss |
| F16 | ~2.4 GB | Reference / further quantization |

### D.5 Verify GGUF locally (llama.cpp)
```bash
llama-cli -m model-Q4_K_M.gguf -p "<|im_start|>user\nनमस्ते, आप कैसे हैं?<|im_end|>\n<|im_start|>assistant\n"
# or LM Studio / Ollama (both support LFM2 GGUFs)
```

### D.6 Why LEAP over our custom scripts?
- Correct LFM2 LoRA module names (`self_attn.out_proj`, `conv.in_proj`, `feed_forward.w1`…)
  — our earlier config used `o_proj`, which matches **zero** modules in LFM2.
- Built-in eval suites (short_answer metric on Hindi val set)
- Checkpoint resume, Ray/Modal/SLURM backends, HF + GGUF export

---

## Timeline Summary

```
Week 1:  Day 1-3  Data collection & validation
         Day 4-5  LCPT (continued pretraining) — optional but recommended
         Day 6-7  LoRA SFT + merging
Week 2:  Day 8-10 Evaluation & benchmarking
         Day 11-14 Bug fixes, hyperparameter tuning, final push to HF Hub
```

**Minimum viable path** (skip LCPT): Days 1-5 (data + SFT).
**Full path** (recommended): 14 days to production-quality model.
