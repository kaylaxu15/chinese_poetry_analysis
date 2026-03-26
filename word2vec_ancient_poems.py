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
from count_ancient_tokens import classical_poems
from count_modern_tokens import modern_poems
from analyze_semantic_shifts import (
    analyze_pairs,
    analyze_nearest_neighbours,
    analyze_pmi_collocates,
    char_tokenize_corpus,
    build_cooccurrence,
)

# ---------------------------------------------------------------------------
# Build corpus: list of token lists (already tokenized in count_modern_tokens)
# ---------------------------------------------------------------------------

ancient_corpus = classical_poems
modern_corpus  = modern_poems
PMI_WINDOW = 3

# ---------------------------------------------------------------------------
# Train Word2Vec
# ---------------------------------------------------------------------------

print("Loading Word2Vec model")
ancient_model = Word2Vec(
    sentences=ancient_corpus,
    vector_size=128,
    window=3,
    min_count=5,
    workers=4,
    sg=1,
    epochs=15,
    compute_loss=True,
)

modern_model = Word2Vec(
    sentences=modern_corpus,
    vector_size=128,
    window=3,
    min_count=5,
    workers=4,
    sg=1,
    epochs=15,
)

# ---------------------------------------------------------------------------
# Save model and word vectors
# ---------------------------------------------------------------------------
ancient_model.save("model/tang_word2vec.model")
ancient_model.wv.save("model/tang_word2vec.wordvectors")
modern_model.save("model/modern_word2vec.model")
modern_model.wv.save("model/modern_word2vec.wordvectors")
print("Models and word vectors saved.")

# ---------------------------------------------------------------------------
# Quick sanity checks
# ---------------------------------------------------------------------------
print(f"Ancient vocabulary size: {len(ancient_model.wv)}")
print(f"Modern vocabulary size:  {len(modern_model.wv)}")

# ---------------------------------------------------------------------------
# Pre-build char corpora and co-occurrence tables once (shared by both files)
# ---------------------------------------------------------------------------
ancient_chars = char_tokenize_corpus(ancient_corpus)
modern_chars  = char_tokenize_corpus(modern_corpus)

anc_uni, anc_cooc, anc_total = build_cooccurrence(ancient_chars, window=PMI_WINDOW)
mod_uni, mod_cooc, mod_total = build_cooccurrence(modern_chars,  window=PMI_WINDOW)
# ---------------------------------------------------------------------------
# File 1: per-word nearest neighbours + PMI collocates
# ---------------------------------------------------------------------------
output_lines = []
analyze_nearest_neighbours(
    ancient_model, modern_model,
    words=["山", "风", "天", "花"],
    topn=10,
    output_lines=output_lines,
)
analyze_pmi_collocates(
    anc_uni=anc_uni, anc_cooc=anc_cooc, anc_total=anc_total,
    mod_uni=mod_uni, mod_cooc=mod_cooc, mod_total=mod_total,
    words=["山", "风", "天", "花"],
    topn=10,
    pmi_window=PMI_WINDOW,
    output_lines=output_lines,
)
with open("analysis_results/words_analysis.txt", "w", encoding="utf-8") as f:
    f.writelines(output_lines)

# ---------------------------------------------------------------------------
# File 2: pair-level cosine + PPMI
# ---------------------------------------------------------------------------
analyze_pairs(
    ancient_model, modern_model,
    anc_uni=anc_uni, anc_cooc=anc_cooc, anc_total=anc_total,
    mod_uni=mod_uni, mod_cooc=mod_cooc, mod_total=mod_total,
    output_path="analysis_results/pairs_analysis.txt",
    pmi_window=PMI_WINDOW,
)