from relational_shifts import cosine_sim, build_auto_anchors
from gensim.models import Word2Vec
from scipy.linalg import orthogonal_procrustes
import numpy as np
import csv

from disappearing_words import compute_disappeared
from count_ancient_tokens import classical_poems
from count_modern_tokens import modern_poems
from analyze_semantic_shifts import is_standard_cjk
from variant_filter import VariantFilter
from word_cloud import plot_top15_equivalents

classical_sentences = classical_poems
modern_sentences    = modern_poems


def find_modern_equivalent(disappeared_ch, model_classical, modern_aligned, topn=5):
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
    # minimum word count = 5
    model_classical = Word2Vec(
        classical_sentences, vector_size=100, window=5, min_count=5, epochs=10
    )
    model_modern = Word2Vec(
        modern_sentences, vector_size=100, window=5, min_count=5, epochs=10
    )

    # --- Procrustes alignment ---
    auto_anchors = build_auto_anchors(model_classical, model_modern)
    A = np.array([model_classical.wv[w] for w in auto_anchors])
    B = np.array([model_modern.wv[w]    for w in auto_anchors])
    Q, _ = orthogonal_procrustes(B, A)
    modern_vectors_aligned = model_modern.wv.vectors @ Q
    modern_aligned = {
        word: modern_vectors_aligned[model_modern.wv.key_to_index[word]]
        for word in model_modern.wv.key_to_index
        if all(is_standard_cjk(ch) for ch in word)
    }

    modern_vocab: set[str] = set(model_modern.wv.key_to_index.keys())

    # --- build variant filter ---
    vf = VariantFilter(
        modern_vocab=modern_vocab,
        modern_aligned=modern_aligned,
        classical_wv=model_classical.wv,
        unihan_path="Unihan_Variants.txt",
        threshold=0.85,
    )

    # --- get ALL disappeared candidates (not just top 20) ---
    results = compute_disappeared(classical_sentences, modern_sentences)
    all_classical = results.get("all_classical", results["top20_classical"])

    # --- classify and report ---
    report = vf.build_report(all_classical)

    # Export full classification report for inspection
    with open("variant_classification_report.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "character", "frequency", "is_variant", "filter_layer",
            "reason", "modern_equivalents"
        ])
        writer.writeheader()
        writer.writerows(report)

    print(f"Full report written to variant_classification_report.csv")

    truly_disappeared = [
        (row["character"], row["frequency"])
        for row in report if not row["is_variant"]
    ]
    discarded = [row for row in report if row["is_variant"]]

    layer_counts = {1: 0, 2: 0, 3: 0}
    for row in discarded:
        if row["filter_layer"] in layer_counts:
            layer_counts[row["filter_layer"]] += 1

    print(f"\nTotal candidates:      {len(all_classical)}")
    print(f"Discarded (variants):  {len(discarded)}")
    print(f"  Layer 1 (Unihan):    {layer_counts[1]}")
    print(f"  Layer 2 (OpenCC/IDS):{layer_counts[2]}")
    print(f"  Layer 3 (embedding): {layer_counts[3]}")
    print(f"Truly disappeared:     {len(truly_disappeared)}")

    # --- find equivalents for the survivors ---
    print("\n=== Modern functional equivalents of disappeared classical characters ===\n")
    for ch, freq in truly_disappeared:
        equivalents = find_modern_equivalent(ch, model_classical, modern_aligned)
        if equivalents is None:
            continue
        equiv_str = "  ".join([f"{w}({s:.3f})" for w, s in equivalents])
        print(f"  {ch} (freq {freq}): {equiv_str}")

    # --- radial collocate word clouds for top 10 truly disappeared ---
    print("\nGenerating radial equivalent plots...")
    plot_top15_equivalents(truly_disappeared, find_modern_equivalent,
                        model_classical, modern_aligned)