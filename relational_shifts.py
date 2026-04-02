from gensim.models import Word2Vec
from scipy.linalg import orthogonal_procrustes
import numpy as np
import os, glob, json, re
import opencc
import opencc
from count_ancient_tokens import sampled_poems as classical_sentences
from count_modern_tokens import unique_poems_data as modern_sentences

converter = opencc.OpenCC('t2s')

def tokenize_ancient(text):
    tokens = list(text)
    # bert tokenizers add [CLS], [SEP] and ## subword prefixes — strip them
    tokens = [t.replace('##', '') for t in tokens]
    tokens = [t for t in tokens if t not in ('[CLS]', '[SEP]', '[UNK]', '[PAD]')]
    return tokens

def cosine_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def build_auto_anchors(model_classical, model_modern,
                       min_count_classical=50, min_count_modern=20,
                       max_anchors=500):
    """
    Automatically selects anchor words that:
    1. Appear in both vocabularies
    2. Are frequent enough in both corpora to have reliable embeddings
    3. Excludes words that shifted a lot (bootstrapped filtering)
    """
    # Get frequent words from each model
    classical_vocab = {
        w for w in model_classical.wv.key_to_index
        if model_classical.wv.get_vecattr(w, "count") >= min_count_classical
    }
    modern_vocab = {
        w for w in model_modern.wv.key_to_index
        if model_modern.wv.get_vecattr(w, "count") >= min_count_modern
    }

    shared = classical_vocab & modern_vocab
    print(f"Shared frequent vocab: {len(shared)} words")

    # First pass: rough alignment on ALL shared words
    shared_list = list(shared)
    A = np.array([model_classical.wv[w] for w in shared_list])
    B = np.array([model_modern.wv[w]    for w in shared_list])
    Q_rough, _ = orthogonal_procrustes(B, A)
    B_aligned_rough = B @ Q_rough

    # Second pass: keep only words whose vectors are close after rough alignment
    # These are the genuinely stable words — good anchors
    stabilities = []
    for i, w in enumerate(shared_list):
        sim = cosine_sim(A[i], B_aligned_rough[i])
        stabilities.append((w, sim))

    # Sort by stability — most stable words make the best anchors
    stabilities.sort(key=lambda x: x[1], reverse=True)

    # Take top N most stable, excluding single-char functional words
    # that are too ambiguous (了, 的, 也, 而, 以, 于, 乃, 乎)
    anchors = [
        w for w, sim in stabilities
        if sim > 0.3
    ][:max_anchors]

    print(f"Selected {len(anchors)} stable anchors")
    print(f"Top 20 most stable: {anchors[:20]}")
    print(f"Stability scores: {[round(s,3) for _,s in stabilities[:20]]}")

    ### check the stability
    import matplotlib.pyplot as plt

    # stabilities is a list of (word, similarity) sorted descending
    top_n = 500
    if len(stabilities) >= top_n:
        cutoff_sim = stabilities[top_n - 1][1]  # cosine similarity of the 500th anchor
    else:
        cutoff_sim = stabilities[-1][1]  # if fewer than 500, take the last one

    stabilities_only = [sim for _, sim in stabilities]

    plt.hist(stabilities_only, bins=50, color='skyblue', edgecolor='black')
    plt.axvline(cutoff_sim, color='red', linestyle='--', label=f'500th anchor cutoff ({cutoff_sim:.3f})')
    plt.xlabel("Cosine similarity (stability)")
    plt.ylabel("Count")
    plt.title("Anchor Stability Histogram")
    plt.legend()
    plt.show()

    return anchors

# Updated functions using aligned space 
def cosine_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def relation_vector(model, a, b):
    return model.wv[b] - model.wv[a]

def relation_shift(a, b):
    """Cosine similarity between relation vectors across aligned spaces."""
    if a not in model_classical.wv or b not in model_classical.wv:
        print(f"'{a}' or '{b}' not in classical vocab"); return None
    if a not in modern_aligned or b not in modern_aligned:
        print(f"'{a}' or '{b}' not in modern vocab"); return None

    r_classical = relation_vector(model_classical, a, b)
    r_modern    = modern_aligned[b] - modern_aligned[a]
    return cosine_sim(r_classical, r_modern)

def shifted_neighbors(a, b, topn=10):
    """Classical relation vector projected into aligned modern space."""
    if a not in model_classical.wv or b not in model_classical.wv:
        print(f"'{a}' or '{b}' not in classical vocab"); return []

    relation = relation_vector(model_classical, a, b)
    scores = {
        word: cosine_sim(relation, vec)
        for word, vec in modern_aligned.items()
    }
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:topn]

def word_shift(word):
    """How much has a single word's meaning shifted across eras?"""
    if word not in model_classical.wv or word not in modern_aligned:
        print(f"'{word}' not in both vocabularies"); return None
    return cosine_sim(model_classical.wv[word], modern_aligned[word])

if __name__ == "__main__":
    # --- Load and train ---

    print(f"Classical poems: {len(classical_sentences)}")
    print(f"Modern poems:    {len(modern_sentences)}")

    model_classical = Word2Vec(classical_sentences, vector_size=100, window=5, min_count=5, epochs=10)
    model_modern    = Word2Vec(modern_sentences,    vector_size=100, window=5, min_count=5, epochs=10)

    # --- Sanity check: within-model neighbors ---
    print("\nClassical neighbors of 月:")
    print(model_classical.wv.most_similar('月', topn=8))
    print("\nModern neighbors of 月:")
    print(model_modern.wv.most_similar('月', topn=8))

    # --- Procrustes alignment ---
    auto_anchors = build_auto_anchors(model_classical, model_modern)

    A = np.array([model_classical.wv[w] for w in auto_anchors])
    B = np.array([model_modern.wv[w]    for w in auto_anchors])
    Q, _ = orthogonal_procrustes(B, A)

    modern_vectors_aligned = model_modern.wv.vectors @ Q
    modern_aligned = {
        word: modern_vectors_aligned[model_modern.wv.key_to_index[word]]
        for word in model_modern.wv.key_to_index
    }
        
    # --- Results ---
    pairs = [('人', '月'), ('人', '梦'), ('人', '花'), ('天', '涯'), ('人', '间')]

    print("\n--- Relation shift scores (1.0 = no change, 0.0 = fully restructured) ---")
    for a, b in pairs:
        score = relation_shift(a, b)
        if score is not None:
            print(f"  {a}→{b}: {score:.4f}")

    print("\n--- Shifted neighbors (人→月 classical relation, projected into modern) ---")
    for word, score in shifted_neighbors('人', '月'):
        print(f"  {word}: {score:.4f}")

    print("\n--- Individual word meaning shift ---")
    for word in ['听', '月', '说', '无', '恨', '已', '问', '上', '梦', '年', '冷', '则', '送', '雨', '谁', '映', '与', '信', '住', '中']: 
        score = word_shift(word)
        if score is not None:
            print(f"  {word}: {score:.4f}  ({'stable' if score > 0.7 else 'shifted'})")