import os
import json
import glob
import re
import random
from collections import Counter
from transformers import AutoTokenizer
from hanziconv import HanziConv

def clean(text):
    text = re.sub(r'[a-zA-Z0-9\s。，、「」！《》？（(）)・；［ ］：/"]', '', text)
    text = HanziConv.toSimplified(text)
    return text

tokenizer = AutoTokenizer.from_pretrained("Jihuai/bert-ancient-chinese")

folder_path = "tang_poems"
file_pattern = os.path.join(folder_path, "poet.tang.*.json")
files = glob.glob(file_pattern)

seen_poems = set()
total_poems = 0
duplicate_count = 0
unique_poems_data = []  # Store (poem_text, tokens) for downsampling

for file in files:
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
        if isinstance(data, list):
            for poem in data:
                total_poems += 1
                paragraphs = poem.get("paragraphs", [])
                poem_text = "".join(paragraphs).strip()
                poem_text = clean(poem_text)

                if poem_text in seen_poems:
                    duplicate_count += 1
                else:
                    seen_poems.add(poem_text)
                    tokens = tokenizer.tokenize(poem_text)
                    unique_poems_data.append((poem_text, tokens))

total_tokens = sum(len(tokens) for _, tokens in unique_poems_data)
print(f"Total poems checked: {total_poems}")
print(f"Duplicate poems found: {duplicate_count}")
print(f"Unique poems: {len(unique_poems_data)}")
print(f"Total tokens (before downsampling): {total_tokens}")

# --- Downsampling ---
TARGET_TOKENS = 2005060

if total_tokens > TARGET_TOKENS:
    random.seed(42)
    random.shuffle(unique_poems_data)

    sampled_poems = []
    running_total = 0
    for poem_text, tokens in unique_poems_data:
        if running_total + len(tokens) > TARGET_TOKENS:
            break
        sampled_poems.append((poem_text, tokens))
        running_total += len(tokens)
else:
    sampled_poems = unique_poems_data
    running_total = total_tokens

print(f"\nAfter downsampling:")
print(f"Sampled poems: {len(sampled_poems)}")
print(f"Total tokens: {running_total}")

# Token stats on downsampled set
token_counter = Counter()
for _, tokens in sampled_poems:
    token_counter.update(tokens)

most_common_tokens = token_counter.most_common(30)
print(f"Total unique tokens: {len(token_counter)}")
print("Top 30 most common tokens:")
for token, count in most_common_tokens:
    print(f"{token}: {count}")