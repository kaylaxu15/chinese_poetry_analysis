import os
import json
import glob
import re
import jieba
from collections import Counter
from hanziconv import HanziConv

# clean special characters
def clean(text):
    text = re.sub(r'[a-zA-Z0-9\s。，、「」！《》？（(）)・；［ ］：/"]', '', text)

    text = HanziConv.toSimplified(text)

    return text

# Use jieba for tokenization
def tokenize(text):
    return list(jieba.cut(text))

# Recursively match ALL JSON files inside modern-poetry and subfolders
file_pattern = os.path.join("modern-poetry", "**", "*.json")
files = glob.glob(file_pattern, recursive=True)

unique_poems_data = []
seen_poems = set()
total_poems = 0
duplicate_count = 0
token_counter = Counter()

for file in files:
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)

        if isinstance(data, list):
            for poem in data:
                total_poems += 1
                paragraphs = poem.get("paragraphs", [])
                poem_text = "".join(paragraphs).strip()

                # remove punctuation and special chars
                poem_text = clean(poem_text)

                # Deduplicate by full text
                if poem_text in seen_poems:
                    duplicate_count += 1
                else:
                    seen_poems.add(poem_text)

                    # Tokenize unique poems and update counter
                    tokens = tokenize(poem_text)
                    token_counter.update(tokens)
                    unique_poems_data.append((poem_text, tokens))

# Get the 10 most common tokens
most_common_tokens = token_counter.most_common(100)

print(f"Total modern poems checked: {total_poems}")
print(f"Duplicate poems found: {duplicate_count}")
print(f"Unique modern poems: {total_poems - duplicate_count}")

# token counting
print(f"Total tokens: {sum(token_counter.values())}")
print(f"Total unique tokens: {len(token_counter)}")
print(f"Top {len(most_common_tokens)} most common tokens (with counts):")

for token, count in most_common_tokens:
    print(f"{token}: {count}")

print([token for token, count in most_common_tokens])
print("Flower count: ", token_counter["花"])

