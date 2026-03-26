import os
import json
import glob
import re
from collections import Counter
from collections import Counter
import opencc
converter = opencc.OpenCC('t2s')

STOP_WORDS = set('的一是不了在我有他这就和你对为之而及且其上来无里何') # most common 25 functional words

def clean(text):
    text = re.sub(r'【.*?】', '', text)  # strip 【京本作X】 style notes
    text = re.sub(r'〔.*?〕', '', text)  # strip 〔...〕 variant brackets
    text = re.sub(r'（.*?）', '', text)  # strip （...） editorial parentheticals
    text = re.sub(r'\(.*?\)', '', text)  # strip ASCII parens too
    text = re.sub(r'[a-zA-Z0-9\s]', '', text)
    # Remove specific punctuation and symbols
    text = re.sub(r'[。，、「」！《》？・；：/～●〇&＝／﹒＋"\-\[\]]', '', text)
    text = converter.convert(text)
    text = ''.join(ch for ch in text if ch not in STOP_WORDS)
    return text

# Use jieba for tokenization
def tokenize(text):
    return list(text)

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

modern_poems = [tokens for _, tokens in unique_poems_data]
# Get the 10 most common tokens
most_common_tokens = token_counter.most_common(100)

if __name__ == "__main__":
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

