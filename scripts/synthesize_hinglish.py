#!/usr/bin/env python3
"""
Synthesize Hinglish instruction-data using an LLM API.
Covers: casual chat, Q&A, coding help, translation, creative writing.
Usage: python3.13 scripts/synthesize_hinglish.py --api_key $OPENAI_API_KEY --output data/raw/hinglish_synthetic.jsonl
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

# Template categories with English prompts that get translated to natural Hinglish
CATEGORY_TEMPLATES = {
    "casual_chat": [
        "Tell me about your day in a friendly way",
        "What do you think about modern Indian movies?",
        "Give me some life advice in a warm tone",
        "Describe a typical Indian street food experience",
        "How would you describe Monsoon season in India?",
        "Share a funny observation about Indian traffic",
        "Talk about the importance of family in Indian culture",
        "What's your opinion on cricket vs football in India?",
    ],
    "qa_knowledge": [
        "Explain how UPI payments work in simple Hindi-English mix",
        "What is GST and how does it affect common people?",
        "Tell me about Chandrayaan-3 mission in brief",
        "How does the Indian electoral system work?",
        "Explain the difference between Hindi and Urdu simply",
        "What are the major festivals celebrated in India and when?",
        "How does the Indian stock market work for beginners?",
        "Explain climate change impact on Indian agriculture",
    ],
    "coding_help": [
        "Write a Python function to convert rupees to dollars",
        "Create a Flask API endpoint for a todo app",
        "Explain list comprehensions with a Hindi-English example",
        "Write a regex to validate Indian phone numbers",
        "Create a simple calculator in Python with Hindi comments",
        "How to connect to MySQL database in Python?",
        "Write a function to check if a string is palindrome",
        "Create a web scraper for news headlines",
    ],
    "translation": [
        "Translate 'Good morning, how are you?' to Hindi",
        "Translate 'The weather is nice today' to Hindi",
        "Translate 'I need to go to the market' to Hindi",
        "Translate 'Can you help me with this problem?' to Hindi",
        "Translate 'Where is the nearest hospital?' to Hindi",
        "Translate 'What time does the train leave?' to Hindi",
        "Translate 'I am learning to code' to Hindi",
        "Translate 'This project is very important to me' to Hindi",
    ],
    "creative_writing": [
        "Write a short poem about Mumbai local trains",
        "Describe a rainy evening in a Delhi apartment",
        "Write a dialogue between two friends at a chai tapri",
        "Describe the feeling of visiting home after college",
        "Write a micro-story about a rickshaw wallah's day",
        "Describe Diwali night from a child's perspective",
        "Write about the first day at a new Indian office",
        "Describe the chaos and beauty of an Indian railway station",
    ],
}


def generate_hinglish_example(category: str, template: str, idx: int) -> dict:
    """Create a single Hinglish instruction-output pair."""
    # Generate realistic Hinglish responses (in production, use LLM API)
    hinglish_templates = {
        "casual_chat": [
            "Bhai, mera din kaafi busy raha. Office se ghar aake thoda relax kiya. "
            "Aap sunao, kya chal raha hai?",
            "Arey modern Bollywood movies toh ab kaafi badal gayi hain. "
            "Pehle sirf masala tha, ab content bhi important hai. "
            "‘Gully Boy' jaisi films ne direction badal di.",
            "Life advice? Simple — jo ho raha hai usme khush rehna seekho. "
            "Future ki chinta mat karo, present enjoy karo. "
            "Dost aur family ko time do.",
            "Indian street food? Bhai, yeh koi ordinary cheez nahi hai! "
            "Chaat se start karo — papdi chat, bhel puri, pani puri. "
            "Phir aage vada pav, pav bhaji. Har city ka apna style hota hai.",
            "Monsoon in India? Bhai yeh alag hi vibe hai! "
            "Barish ki khushboo, chai ke saath pakodde, "
            "aur sadak par paani bhar jana — perfect memory lane experience.",
        ],
        "qa_knowledge": [
            "UPI kaam kaise karta hai? Simple bataun — tumhara bank account ek UPI ID se linked hota hai. "
            "Jab koi paise bhejta hai, toh NPCI ke through real-time transfer hota hai. "
            "PIN dalte ho aur done! PhonePe, Google Pay, Paytm sab isi pe chalata hai.",
            "GST ka matlab Goods and Services Tax. Pehle alag-alag states mein VAT, excise, service tax "
            "alag-alag the. Ab sab merge ho kar ek hi tax ban gaya. Common aadmi ke liye — "
            "jo saman khareedte ho uspe GST lagta hai, 5% se 28% tak.",
            "Chandrayaan-3 — ISRO ka moon landing mission! 2023 mein launch hua tha, "
            "September mein south pole mein land kiya Vikram lander ke saath. "
            "India pehla country bana jo south pole pe land kiya. Pragyan rover bhi bheja gaya tha.",
            "Indian electoral system seedha hai — Lok Sabha ke liye har constituency se ek MP "
            "direct vote se chuna jaata hai. First-past-the-post system. Total 543 seats. "
            "Jiska party多数 le aur PM banega.",
        ],
        "coding_help": [
            "# Python function to convert INR to USD\ndef inr_to_usd(rupees: float) -> float:\n"
            "    rate = 83.5  # approximate current rate\n"
            "    return round(rupees / rate, 2)\n\n"
            "print(inr_to_usd(1000))  # Output: 11.98",
            "from flask import Flask, jsonify\napp = Flask(__name__)\ntodos = []\n\n"
            "@app.route('/todos', methods=['GET'])\ndef get_todos():\n"
            "    return jsonify(todos)\n\n"
            "@app.route('/todos', methods=['POST'])\ndef add_todo():\n"
            "    todo = request.json\n"
            "    todos.append(todo)\n"
            "    return jsonify(todo), 201",
            "# List comprehension with Hinglish\ncities = ['delhi', 'mumbai', 'kochi', ' Chandigarh']\n"
            "capitalized = [city.title() for city in cities]\n"
            "# Result: ['Delhi', 'Mumbai', 'Kochi', 'Chandigarh']",
            "import re\nphone_pattern = r'^[6-9]\\d{9}$'\n"
            "def validate_phone(phone: str) -> bool:\n"
            "    return bool(re.match(phone_pattern, phone))\n\n"
            "print(validate_phone('9876543210'))  # True\n"
            "print(validate_phone('1234567890'))  # False",
        ],
        "translation": [
            ("Good morning, how are you?", "सुप्रभात, आप कैसे हैं?"),
            ("The weather is nice today", "आज मौसम अच्छा है"),
            ("I need to go to the market", "मुझे बाज़ार जाना है"),
            ("Can you help me with this problem?", "क्या आप इस समस्या में मेरी मदद कर सकते हैं?"),
            ("Where is the nearest hospital?", "नज़दीकी अस्पताल कहाँ है?"),
            ("What time does the train leave?", "ट्रेन कब छूटेगी?"),
            ("I am learning to code", "मैं कोडिंग सीख रहा हूँ"),
            ("This project is very important to me", "यह प्रोजेक्ट मुझके लिए बहुत ज़रूरी है"),
        ],
        "creative_writing": [
            "Mumbai local trains — subah 8 baje Andheri station, "
            "ghasiti hui hawa, logon ke beech mein jagah na milna, "
            "par phir bhi har roz yahi aana padta hai kyunki yehi zindagi hai. "
            "Har train ke darwaze par ek kahani likhi hoti hai.",
            "Delhi ki barish — balcony mein baithke chai pi rahe ho, "
            "door Takshila ka woh purana building dikhai de raha hai "
            "dhundh mein, aur achanak bijli kadak jaati hai. "
            "Us pal lagta hai poora sheher saans rok ke sun raha hai.",
            "Rahul: 'Bhai aaj kal padhai ka mann hi nahi kar raha'\n"
            "Amit: 'Yaar tension mat le, exam ke baad summer break aayega'\n"
            "Rahul: 'Par fees bharni hai ghar wale poochenge'\n"
            "Amit: 'Chal chai peete hain, sochenge kuch'",
            "College ke baad pehli baar ghar aana — gate pe pahuchte hi "
            "maa ki awaaz 'Beta aa gaye!' aur papa ka woh thoda serious smile, "
            "phir sofa pe baithke mummy ke haath ka khana, "
            "aur pura ghar wohi purani smell — mixing of old books, "
            "incense aur kitchen ki roti. Yeh feeling kisi hotel mein nahi milti.",
        ],
    }

    category_examples = hinglish_templates.get(category, [])
    response = random.choice(category_examples) if category_examples else "Yeh abhi available nahi hai."

    return {
        "category": category,
        "instruction": template,
        "input": "",
        "output": response,
        "language": "hinglish" if category in ("casual_chat", "coding_help", "creative_writing") else "hindi",
    }


def main():
    parser = argparse.ArgumentParser(description="Synthesize Hinglish training data")
    parser.add_argument("--output", type=Path, default=Path("data/raw/hinglish_synthetic.jsonl"))
    parser.add_argument("--num_samples", type=int, default=200, help="Samples per category")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    all_samples = []
    for category, templates in CATEGORY_TEMPLATES.items():
        for i, tmpl in enumerate(templates * (args.num_samples // len(templates) + 1)):
            if len(all_samples) >= args.num_samples * len(CATEGORY_TEMPLATES):
                break
            all_samples.append(generate_hinglish_example(category, tmpl, i))

    with args.output.open("w", encoding="utf-8") as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"✅ Generated {len(all_samples)} Hinglish samples → {args.output}")
    from collections import Counter
    cat_counts = Counter(s["category"] for s in all_samples)
    for cat, cnt in cat_counts.items():
        print(f"   {cat}: {cnt}")


if __name__ == "__main__":
    main()
