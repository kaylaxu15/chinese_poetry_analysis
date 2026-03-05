"""
train_word2vec.py

Trains Word2Vec embeddings on Tang poetry using the tokenized,
deduplicated, and downsampled data from count_modern_tokens.py.

Usage:
    python train_word2vec.py
"""

import logging
from gensim.models import Word2Vec

# Import sampled poems and tokenizer 
from count_ancient_tokens import sampled_poems
from count_modern_tokens import unique_poems_data
from analyze_semantic_shifts import analyze_pairs

# ---------------------------------------------------------------------------
# Build corpus: list of token lists (already tokenized in count_modern_tokens)
# ---------------------------------------------------------------------------

ancient_corpus = [tokens for _, tokens in sampled_poems]
modern_corpus = [tokens for _, tokens in unique_poems_data]

# -----------------------------------------
# Train Word2Vec
# -----------------------------------------

print("Loading Word2Vec model")
ancient_model = Word2Vec(
    sentences=ancient_corpus,
    vector_size=128,
    window=5,
    min_count=2,
    workers=4,
    sg=1,
    epochs=10,
    compute_loss=True,
)

modern_model = Word2Vec(
    sentences=modern_corpus,
    vector_size=128,
    window=5,
    min_count=2,
    workers=4,
    sg=1,
    epochs=10,
)

# --------------------------------------------------
# Save model and word vectors (fixed paths)
# --------------------------------------------------
ancient_model.save("model/tang_word2vec.model")
ancient_model.wv.save("model/tang_word2vec.wordvectors")
modern_model.save("model/modern_word2vec.model")
modern_model.wv.save("model/modern_word2vec.wordvectors")
print("Models and word vectors saved.")

# -----------------------------------------------------
# Quick sanity checks
# -----------------------------------------------------
print(f"Ancient vocabulary size: {len(ancient_model.wv)}")
print(f"Modern vocabulary size:  {len(modern_model.wv)}")

# -----------------------------------------------------
# Side-by-side nearest neighbour comparison
# -----------------------------------------------------
words_of_interest = ["有", "人", "不"]
TOPN = 10
collocates_lines = ["Top 10 Collocates with Cosine Similarity\n", "=" * 60 + "\n\n"]

print("\n--- Nearest neighbours (cosine similarity) ---\n")

for word in words_of_interest:
    # Ancient neighbours
    if word in ancient_model.wv:
        ancient_nbrs = ancient_model.wv.most_similar(word, topn=TOPN)
        ancient_str = ", ".join(f"{w}({s:.3f})" for w, s in ancient_nbrs)
    else:
        ancient_str = "(not in vocab)"

    # Modern neighbours
    if word in modern_model.wv:
        modern_nbrs = modern_model.wv.most_similar(word, topn=TOPN)
        modern_str = ", ".join(f"{w}({s:.3f})" for w, s in modern_nbrs)
    else:
        modern_str = "(not in vocab)"

    print(f"Word: {word}")
    print(f"  TANG:   {ancient_str}")
    print(f"  MODERN: {modern_str}")
    print()

    collocates_lines.append(f"Word: {word}\n")
    collocates_lines.append(f"  TANG:   {ancient_str}\n")
    collocates_lines.append(f"  MODERN: {modern_str}\n\n")

with open("analysis_results/top_10_collocates.txt", "w", encoding="utf-8") as f:
    f.writelines(collocates_lines)


# -----------------------------------------------------
# Side-by-side analysis of word pair relationships
# -----------------------------------------------------

analyze_pairs(ancient_model, modern_model, output_path="analysis_results/romanticized_relationships.txt")