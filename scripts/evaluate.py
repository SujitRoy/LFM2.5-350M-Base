#!/usr/bin/env python3
"""
Evaluation script for LFM2.5-350M-Hindi.
Run on GPU after training. Tests Hindi, Hinglish, and English retention.
Usage: python3.13 scripts/evaluate.py --model_path output/lfm25-350m-hindi-merged
"""

import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ── Evaluation prompts ──────────────────────────────────────────────
HINDI_TESTS = [
    ("नमस्ते! आप कैसे हैं?", "Greeting / well-being check"),
    ("मुझे हिंदी में मदद चाहिए", "Request for help in Hindi"),
    ("आप मुझे एक कहानी सुनाइए", "Tell a story in Hindi"),
    ("भारत की राजधानी क्या है?", "Factual QA in Hindi"),
    ("2+2 का जवाब हिंदी में दें", "Simple math in Hindi"),
    ("मौसम के बारे में बताएं", "Weather description"),
    ("एक कविता लिखें बरसात पर", "Creative writing prompt"),
    ("UPI क्या होती है? समझाइए", "Explain UPI in Hindi"),
]

HINGLISH_TESTS = [
    ("Bhai yeh code kaise kaam karta hai?", "Code explanation in Hinglish"),
    ("Mujhe Hindi seekhni hai, kya karun?", "Learning advice in Hinglish"),
    ("Aaj kal ke trending movies kya hain?", "Pop culture Q&A"),
    ("Python mein loop kaise likhte hain?", "Coding help in Hinglish"),
    ("Meri life bahut stressful hai, kya karun?", "Emotional advice"),
    ("React vs Angular kaunsa better hai?", "Tech comparison"),
]

ENGLISH_RETENTION_TESTS = [
    ("What is the capital of France?", "Capital city QA"),
    ("Write a Python function to reverse a string.", "Coding"),
    ("Explain quantum entanglement simply.", "Science explanation"),
    ("List 5 famous Indian dishes with descriptions.", "Cross-lingual knowledge"),
]


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 200) -> str:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7, top_p=0.9)
    generated = tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    return generated.strip()


def has_hindi_script(text: str) -> bool:
    return bool(re.search(r'[\u0900-\u097F]', text))


def evaluate(model_path: str, num_tests: int = 3):
    print(f"Loading model from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).cuda()
    model.eval()

    results = {"hindi": [], "hinglish": [], "english_retention": []}

    print("\n" + "=" * 60)
    print("HINDI EVALUATION")
    print("=" * 60)
    for prompt, desc in HINDI_TESTS[:num_tests]:
        ans = generate(model, tokenizer, prompt)
        has_devanagari = has_hindi_script(ans)
        flag = "✅" if has_devanagari else "⚠️  NO HINDI SCRIPT"
        print(f"\n[{flag}] {desc}")
        print(f"  Q: {prompt}")
        print(f"  A: {ans[:300]}{'...' if len(ans) > 300 else ''}")
        results["hindi"].append({"prompt": prompt, "response": ans, "has_hindi_script": has_devanagari})

    print("\n" + "=" * 60)
    print("HINGLISH EVALUATION")
    print("=" * 60)
    for prompt, desc in HINGLISH_TESTS[:num_tests]:
        ans = generate(model, tokenizer, prompt)
        mixed = has_hindi_script(ans) and any(w.isascii() for w in ans.split())
        flag = "✅" if mixed else "⚠️  NO CODE-MIX"
        print(f"\n[{flag}] {desc}")
        print(f"  Q: {prompt}")
        print(f"  A: {ans[:300]}{'...' if len(ans) > 300 else ''}")
        results["hinglish"].append({"prompt": prompt, "response": ans, "has_mixed": mixed})

    print("\n" + "=" * 60)
    print("ENGLISH RETENTION (should still work)")
    print("=" * 60)
    for prompt, desc in ENGLISH_RETENTION_TESTS[:num_tests]:
        ans = generate(model, tokenizer, prompt)
        has_eng = bool(re.search(r'[a-zA-Z]{3,}', ans))
        flag = "✅" if has_eng else "⚠️  NO ENGLISH"
        print(f"\n[{flag}] {desc}")
        print(f"  Q: {prompt}")
        print(f"  A: {ans[:300]}{'...' if len(ans) > 300 else ''}")
        results["english_retention"].append({"prompt": prompt, "response": ans, "has_english": has_eng})

    # Save results
    out_path = Path("logs/evaluation_results.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n📊 Results saved → {out_path}")

    # Summary
    hindi_ok = sum(r["has_hindi_script"] for r in results["hindi"])
    hinglish_ok = sum(r["has_mixed"] for r in results["hinglish"])
    english_ok = sum(r["has_english"] for r in results["english_retention"])
    print(f"\n📈 Summary: Hindi ✅{hindi_ok}/{num_tests} | Hinglish ✅{hinglish_ok}/{num_tests} | English retention ✅{english_ok}/{num_tests}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="output/lfm25-350m-hindi-merged")
    parser.add_argument("--num_tests", type=int, default=3)
    args = parser.parse_args()
    evaluate(args.model_path, args.num_tests)
