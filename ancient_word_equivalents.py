from relational_shifts import get_poems, cosine_sim, build_auto_anchors
from gensim.models import Word2Vec
from scipy.linalg import orthogonal_procrustes
import numpy as np
from disappearing_words import compute_disappeared

classical_sentences = get_poems(True)
modern_sentences    = get_poems(False)

def find_modern_equivalent(disappeared_ch, model_classical, modern_aligned, topn=5):
    """For a disappeared classical character, find its closest modern equivalents."""
    if disappeared_ch not in model_classical.wv:
        return None
    classical_vec = model_classical.wv[disappeared_ch]
    scores = {
        word: cosine_sim(classical_vec, vec)
        for word, vec in modern_aligned.items()
    }
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:topn]

if __name__ == "__main__":
    # --- train models ---
    model_classical = Word2Vec(classical_sentences, vector_size=100, window=5, min_count=5, epochs=10)
    model_modern = Word2Vec(modern_sentences, vector_size=100, window=5, min_count=5, epochs=10)

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

    # --- disappeared words ---
    results = compute_disappeared(classical_sentences, modern_sentences)
    top20_classical = results["top20_classical"]
    top20_modern    = results["top20_modern"]

    # --- find equivalents ---
    print("=== Modern functional equivalents of disappeared classical characters ===\n")
    for ch, freq in top20_classical:
        equivalents = find_modern_equivalent(ch, model_classical, modern_aligned)
        if equivalents is None:
            print(f"  {ch} (freq {freq}): not in Word2Vec vocab (too rare)")
            continue
        equiv_str = "  ".join([f"{w}({s:.3f})" for w, s in equivalents])
        print(f"  {ch} (freq {freq}): {equiv_str}")

    print("\n=== Classical functional equivalents of disappeared modern characters ===\n")
    for ch, freq in top20_modern:
        if ch not in model_modern.wv:
            print(f"  {ch} (freq {freq}): not in Word2Vec vocab (too rare)")
            continue
        modern_vec = modern_aligned[ch]
        scores = {
            word: cosine_sim(modern_vec, model_classical.wv[word])
            for word in model_classical.wv.key_to_index
        }
        top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
        equiv_str = "  ".join([f"{w}({s:.3f})" for w, s in top])
        print(f"  {ch} (freq {freq}): {equiv_str}")