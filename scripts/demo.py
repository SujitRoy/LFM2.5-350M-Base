#!/usr/bin/env python3
"""
Quick inference demo for the fine-tuned LFM2.5-350M-Hindi model.
Usage:
  python3.13 scripts/demo.py --model_path output/lfm25-350m-hindi-merged
  # or use LoRA adapter:
  python3.13 scripts/demo.py --base LiquidAI/LFM2.5-350M-Base --adapter output/sft_lora/adapter
"""

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PeftModel


PROMPTS = [
    "नमस्ते! मुझे हिंदी मendum सीखना हें।",
    "Bhai, mujhe Python sikha do basics se.",
    "India ke baare mein batao.",
    "Good morning! Aaj kya plan hai?",
    "Ek choti si kahani likho mountain ke baare mein.",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--base", type=str, default="LiquidAI/LFM2.5-350M-Base")
    parser.add_argument("--adapter", type=str, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)

    if args.adapter:
        print(f"Loading base: {args.base}")
        model = AutoModelForCausalLM.from_pretrained(
            args.base, torch_dtype=torch.bfloat16, trust_remote_code=True
        ).to(device)
        model = PeftModel.from_pretrained(model, args.adapter)
        print(f"Loaded LoRA adapter: {args.adapter}")
    elif args.model_path:
        print(f"Loading merged model: {args.model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
        ).to(device)
    else:
        print(f"Loading base model: {args.base}")
        model = AutoModelForCausalLM.from_pretrained(
            args.base, torch_dtype=torch.bfloat16, trust_remote_code=True
        ).to(device)

    model.eval()

    for prompt in PROMPTS:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=200, do_sample=True,
                temperature=0.7, top_p=0.9,
            )
        response = tokenizer.decode(
            out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
        )
        print(f"\n{'='*50}")
        print(f"Q: {prompt}")
        print(f"A: {response}")


if __name__ == "__main__":
    main()
