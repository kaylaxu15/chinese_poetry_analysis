import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from collections import Counter
from count_ancient_tokens import classical_poems
from count_modern_tokens import modern_poems
import numpy as np

cjk_font = fm.FontProperties(fname="/System/Library/Fonts/STHeiti Light.ttc")
fm.fontManager.addfont("/System/Library/Fonts/STHeiti Light.ttc")

# ── load corpora ──────────────────────────────────────────────────────────────

def is_standard_cjk(ch):
    return '\u4e00' <= ch <= '\u9fff'

#---'Collocate functions-----
def build_bigrams(sentences):
    bigrams = Counter()
    for sent in sentences:
        for i in range(len(sent) - 1):
            bigrams[(sent[i], sent[i+1])] += 1
    return bigrams

def top_collocates(ch, bigrams, n=8):
    collocates = Counter()
    for (a, b), count in bigrams.items():
        if a == ch:
            collocates[b] += count
        elif b == ch:
            collocates[a] += count
    return collocates.most_common(n)

def plot_collocate_heatmap(top20, bigrams, all_chars, color, title, filename):
    # Collect top collocates for each disappeared character
    collocate_data = {}
    for ch, freq in top20:
        results = top_collocates(ch, bigrams, n=10)
        collocate_data[ch] = dict(results)

    # Get union of all collocates that appear, ranked by total frequency
    all_collocates = Counter()
    for ch_dict in collocate_data.values():
        all_collocates.update(ch_dict)
    top_collocate_words = [w for w, _ in all_collocates.most_common(20)]

    # Build matrix: rows = disappeared chars, cols = collocate words
    disappeared_chars = [ch for ch, _ in top20]
    matrix = np.zeros((len(disappeared_chars), len(top_collocate_words)))
    for i, ch in enumerate(disappeared_chars):
        for j, w in enumerate(top_collocate_words):
            matrix[i, j] = collocate_data[ch].get(w, 0)

    # Normalize each row so rare chars are still visible
    row_maxes = matrix.max(axis=1, keepdims=True)
    row_maxes[row_maxes == 0] = 1
    matrix_norm = matrix / row_maxes

    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(matrix_norm, aspect='auto', cmap='Blues' if color == 'blue' else 'Greens')

    ax.set_xticks(range(len(top_collocate_words)))
    ax.set_yticks(range(len(disappeared_chars)))
    ax.set_xticklabels(top_collocate_words, fontproperties=cjk_font, fontsize=12)
    ax.set_yticklabels(
        [f"{ch} ({freq})" for ch, freq in top20],
        fontproperties=cjk_font, fontsize=12
    )

    ax.set_title(title, fontproperties=cjk_font, fontsize=13)
    ax.set_xlabel("Collocate character", fontsize=11)
    ax.set_ylabel("Disappeared character (freq in source corpus)", fontsize=11)

    # Add raw count annotations inside cells
    for i in range(len(disappeared_chars)):
        for j in range(len(top_collocate_words)):
            raw = int(matrix[i, j])
            if raw > 0:
                ax.text(j, i, str(raw), ha='center', va='center',
                        fontsize=7, color='white' if matrix_norm[i,j] > 0.6 else 'black')

    plt.colorbar(im, ax=ax, label='Normalized co-occurrence (row max = 1)')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()
    print(f"Saved: {filename}")

# ── side-by-side Zipf plots ───────────────────────────────────────────────────
def plot_zipf(ax, top50, color, title, xlabel, ylabel):
    chars  = [ch for ch, _ in top50]
    counts = [ct for _, ct in top50]
    ranks  = list(range(1, len(top50) + 1))
    ax.bar(ranks, counts, color=color, width=0.7, alpha=0.85)
    ax.set_xticks(ranks)
    ax.set_xticklabels(chars, fontproperties=cjk_font, fontsize=10)
    ax.set_title(title, fontproperties=cjk_font, fontsize=13)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_yscale('log')
    ax.grid(axis='y', linestyle='--', alpha=0.4)

#---- Computing disappeared words---
def compute_disappeared(classical_poems, modern_poems):
 
    classical_chars = Counter(
        ch for sent in classical_poems
        for ch in sent if is_standard_cjk(ch)
    )
    modern_chars = Counter(
        ch for sent in modern_poems
        for ch in sent if is_standard_cjk(ch)
    )
 
    disappeared_classical = {
        ch: count for ch, count in classical_chars.items()
        if ch not in modern_chars
    }
    disappeared_modern = {
        ch: count for ch, count in modern_chars.items()
        if ch not in classical_chars
    }
 
    disappeared_classical_sorted = sorted(
        disappeared_classical.items(), key=lambda x: x[1], reverse=True
    )
    disappeared_modern_sorted = sorted(
        disappeared_modern.items(), key=lambda x: x[1], reverse=True
    )
 
    return {
        "count_disappeared_classical": len(disappeared_classical),
        "count_disappeared_modern":    len(disappeared_modern),
        "all_classical":               disappeared_classical_sorted,
        "all_modern":                  disappeared_modern_sorted,
        "top20_classical":             disappeared_classical_sorted[:20],
        "top50_classical":             disappeared_classical_sorted[:50],
        "top20_modern":                disappeared_modern_sorted[:20],
        "top50_modern":                disappeared_modern_sorted[:50],
        "classical_chars":             classical_chars,
        "modern_chars":                modern_chars,
    }


if __name__== "__main__":
    results = compute_disappeared(classical_poems, modern_poems)
    top20_classical = results["top20_classical"]
    top20_modern = results["top20_modern"]
    top50_classical = results["top50_classical"]
    top50_modern = results["top50_modern"]
    classical_chars = results["classical_chars"]
    modern_chars = results["modern_chars"]

    # counts
    disappeared_modern_count = results["count_disappeared_modern"]
    disappeared_classical_count = results["count_disappeared_classical"]

    print(f"Total classical-only characters: {disappeared_classical_count}")
    print(f"Total modern-only characters: {disappeared_modern_count}")

    print(f"\nTop 20 classical-only (disappeared from modern):")
    for ch, count in top20_classical:
        print(f"  {ch}: {count}")

    print(f"\nTop 20 modern-only (new in modern, absent from classical):")
    for ch, count in top20_modern:
        print(f"  {ch}: {count}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))

    plot_zipf(ax1, top50_classical,
            color='#7F77DD',
            title=f'Characters Found Only in Ancient Corpus: Total of {disappeared_classical_count}',
            xlabel='Character ranked by classical frequency',
            ylabel='Frequency in classical corpus')

    plot_zipf(ax2, top50_modern,
            color='#1D9E75',
            title=f'Characters Found Only in Modern Corpus: Total of {disappeared_modern_count}',
            xlabel='Character ranked by modern frequency',
            ylabel='Frequency in modern corpus')

    plt.tight_layout()
    plt.savefig("disappeared_zipf_both.png", dpi=150)
    plt.show()
    print("Saved: disappeared_zipf_both.png")

    # ── bigram collocates ─────────────────────────────────────────────────────────

    classical_bigrams = build_bigrams(classical_poems)
    modern_bigrams    = build_bigrams(modern_poems)

    print("\n=== Classical bigram collocates of top 20 classical-only characters ===\n")
    for ch, freq in top20_classical:
        results = top_collocates(ch, classical_bigrams)
        collocate_str = "  ".join([f"{w}({c})" for w, c in results])
        print(f"  {ch} (freq {freq}): {collocate_str}")

    print("\n=== Modern bigram collocates of top 20 modern-only characters ===\n")
    for ch, freq in top20_modern:
        results = top_collocates(ch, modern_bigrams)
        collocate_str = "  ".join([f"{w}({c})" for w, c in results])
        print(f"  {ch} (freq {freq}): {collocate_str}")


    plot_collocate_heatmap(
        top20_classical, classical_bigrams, classical_chars,
        color='blue',
        title='Heatmap of Collocates of Classical-Only Characters',
        filename='collocates_classical.png'
    )

    plot_collocate_heatmap(
        top20_modern, modern_bigrams, modern_chars,
        color='green',
        title='Heatmap of Collocates of Modern-Only Character',
        filename='collocates_modern.png'
    )