from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from rich.console import Console
from rich.table import Table
from rich import box
from collections import defaultdict
import numpy as np
import math

console = Console()

# ---------------------------------------------------------------------------
# Words and pairs of interest
# ---------------------------------------------------------------------------
WORDS_OF_INTEREST = ["山", "风", "天", "花", "人", "月", "梦", "水"]

PAIRS = [
    ("人", "月", "person–moon   (nature/romance)"),
    ("人", "梦", "person–dream  (dream)"),
    ("人", "花", "person–flower (nature/beauty)"),
]

def is_standard_cjk(ch):
    return '\u4e00' <= ch <= '\u9fff'

# ---------------------------------------------------------------------------
# Character-level tokenization
# ---------------------------------------------------------------------------

def char_tokenize_corpus(corpus):
    """Re-tokenize a corpus (list of token lists) into individual characters."""
    char_corpus = []
    for tokens in corpus:
        chars = [ch for token in tokens for ch in token if ch.strip() and is_standard_cjk(ch)]
        if chars:
            char_corpus.append(chars)
    return char_corpus


# ---------------------------------------------------------------------------
# Co-occurrence and PMI
# ---------------------------------------------------------------------------

def build_cooccurrence(char_corpus, window=3):
    unigram_counts = defaultdict(int)
    cooc_counts    = defaultdict(int)
    total_tokens   = 0

    for doc in char_corpus:
        total_tokens += len(doc)
        for i, target in enumerate(doc):
            if not is_standard_cjk(target):   # ← add this
                continue
            unigram_counts[target] += 1
            start = max(0, i - window)
            end   = min(len(doc), i + window + 1)
            for j in range(start, end):
                if j == i:
                    continue
                context = doc[j]
                if not is_standard_cjk(context):  # ← and this
                    continue
                pair = (min(target, context), max(target, context))
                cooc_counts[pair] += 1

    return unigram_counts, cooc_counts, total_tokens


def pmi_score(word1, word2, unigram_counts, cooc_counts, total_tokens, positive=True):
    """Compute (P)PMI for a word pair. Returns None if no co-occurrence."""
    pair  = (min(word1, word2), max(word1, word2))
    cooc  = cooc_counts.get(pair, 0)
    if cooc == 0:
        return None

    total_cooc = sum(cooc_counts.values())
    p_w1   = unigram_counts[word1] / total_tokens
    p_w2   = unigram_counts[word2] / total_tokens
    p_w1w2 = cooc / total_cooc

    if p_w1 == 0 or p_w2 == 0:
        return None

    raw_pmi = math.log2(p_w1w2 / (p_w1 * p_w2))
    return max(0.0, raw_pmi) if positive else raw_pmi


def top_pmi_collocates(word, unigram_counts, cooc_counts, total_tokens, topn=10):
    """Return top-n collocates of word ranked by PPMI score."""
    scores = []
    for (w1, w2) in cooc_counts:
        if w1 == word:
            other = w2
        elif w2 == word:
            other = w1
        else:
            continue

        if not is_standard_cjk(other):  # filter for unknown words
            continue
        score = pmi_score(word, other, unigram_counts, cooc_counts, total_tokens)
        if score is not None:
            scores.append((other, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:topn]


# ---------------------------------------------------------------------------
# Cosine similarity (within a single model)
# ---------------------------------------------------------------------------

def cosine_sim(model, word1, word2):
    """Return cosine similarity between two words in one model, or None if OOV."""
    if word1 not in model.wv or word2 not in model.wv:
        return None, [w for w in (word1, word2) if w not in model.wv]
    v1 = model.wv[word1].reshape(1, -1)
    v2 = model.wv[word2].reshape(1, -1)
    return float(cosine_similarity(v1, v2)[0][0]), []


# ---------------------------------------------------------------------------
# Helper: ensure co-occurrence tables are available
# ---------------------------------------------------------------------------

def _ensure_cooc_tables(
    anc_uni, anc_cooc, anc_total,
    mod_uni, mod_cooc, mod_total,
    ancient_corpus, modern_corpus,
    pmi_window,
):
    """
    Return pre-built tables if all provided, otherwise build from raw corpora.
    Returns (pmi_available, anc_uni, anc_cooc, anc_total, mod_uni, mod_cooc, mod_total).
    """
    if all(x is not None for x in [anc_uni, anc_cooc, anc_total, mod_uni, mod_cooc, mod_total]):
        return True, anc_uni, anc_cooc, anc_total, mod_uni, mod_cooc, mod_total

    if ancient_corpus is not None and modern_corpus is not None:
        ancient_chars = char_tokenize_corpus(ancient_corpus)
        modern_chars  = char_tokenize_corpus(modern_corpus)
        anc_uni, anc_cooc, anc_total = build_cooccurrence(ancient_chars, window=pmi_window)
        mod_uni, mod_cooc, mod_total = build_cooccurrence(modern_chars,  window=pmi_window)
        return True, anc_uni, anc_cooc, anc_total, mod_uni, mod_cooc, mod_total

    return False, None, None, None, None, None, None


# ---------------------------------------------------------------------------
# Analysis 1: within-model nearest neighbours
# ---------------------------------------------------------------------------

def analyze_nearest_neighbours(
    ancient_model,
    modern_model,
    words=None,
    topn=10,
    output_lines=None,
):
    """
    For each word, print top-N most similar words within each model separately.
    No cross-model vector comparison.
    """
    if words is None:
        words = WORDS_OF_INTEREST
    if output_lines is None:
        output_lines = []

    header = "Nearest Neighbours (Word2Vec Cosine Similarity)"
    output_lines.append(f"\n{header}\n" + "=" * 80 + "\n")

    table = Table(
        title=f"\n{header}",
        box=box.ROUNDED, show_lines=True,
        title_style="bold cyan", header_style="bold magenta",
    )
    table.add_column("Word",          style="bold white", justify="center", min_width=6)
    table.add_column("Tang top-10",   style="yellow",     justify="left",   min_width=50)
    table.add_column("Modern top-10", style="green",      justify="left",   min_width=50)

    for word in words:
        anc_nbrs = [(w, s) for w, s in ancient_model.wv.most_similar(word, topn=topn*2)
            if is_standard_cjk(w)][:topn] if word in ancient_model.wv else []
        mod_nbrs = [(w, s) for w, s in modern_model.wv.most_similar(word, topn=topn*2)
                    if is_standard_cjk(w)][:topn] if word in modern_model.wv else []

        anc_str = ", ".join(f"{w}({s:.3f})" for w, s in anc_nbrs) or "(not in vocab)"
        mod_str = ", ".join(f"{w}({s:.3f})" for w, s in mod_nbrs) or "(not in vocab)"

        table.add_row(word, anc_str, mod_str)
        output_lines.append(f"  {word}:\n")
        output_lines.append(f"    Tang:   {anc_str}\n")
        output_lines.append(f"    Modern: {mod_str}\n\n")

    console.print(table)
    return output_lines


# ---------------------------------------------------------------------------
# Analysis 2: within-corpus PMI collocates
# ---------------------------------------------------------------------------

def analyze_pmi_collocates(
    anc_uni=None, anc_cooc=None, anc_total=None,
    mod_uni=None, mod_cooc=None, mod_total=None,
    ancient_corpus=None,
    modern_corpus=None,
    words=None,
    topn=10,
    pmi_window=5,
    output_lines=None,
):
    """
    For each word, print top-N PMI-weighted collocates within each corpus.
    Accepts pre-built tables or raw corpora.
    """
    if words is None:
        words = WORDS_OF_INTEREST
    if output_lines is None:
        output_lines = []

    pmi_available, anc_uni, anc_cooc, anc_total, mod_uni, mod_cooc, mod_total = \
        _ensure_cooc_tables(
            anc_uni, anc_cooc, anc_total,
            mod_uni, mod_cooc, mod_total,
            ancient_corpus, modern_corpus,
            pmi_window,
        )

    if not pmi_available:
        console.print("[red]No corpus data provided for PMI collocates.[/red]")
        return output_lines, None, None, None, None, None, None

    header = f"Top {topn} PMI Collocates (character-level, window={pmi_window})"
    output_lines.append(f"\n{header}\n" + "=" * 80 + "\n")

    table = Table(
        title=f"\n{header}",
        box=box.ROUNDED, show_lines=True,
        title_style="bold cyan", header_style="bold magenta",
    )
    table.add_column("Word",                  style="bold white", justify="center", min_width=6)
    table.add_column("Tang PMI collocates",   style="yellow",     justify="left",   min_width=50)
    table.add_column("Modern PMI collocates", style="green",      justify="left",   min_width=50)

    for word in words:
        anc_col = top_pmi_collocates(word, anc_uni, anc_cooc, anc_total, topn=topn)
        mod_col = top_pmi_collocates(word, mod_uni, mod_cooc, mod_total, topn=topn)
        anc_str = ", ".join(f"{w}({s:.3f})" for w, s in anc_col) or "(none)"
        mod_str = ", ".join(f"{w}({s:.3f})" for w, s in mod_col) or "(none)"
        table.add_row(word, anc_str, mod_str)
        output_lines.append(f"  {word}:\n")
        output_lines.append(f"    Tang PMI:   {anc_str}\n")
        output_lines.append(f"    Modern PMI: {mod_str}\n\n")

    console.print(table)
    return output_lines, anc_uni, anc_cooc, anc_total, mod_uni, mod_cooc, mod_total


# ---------------------------------------------------------------------------
# Analysis 3: pair-level cosine + PPMI
# ---------------------------------------------------------------------------

def analyze_pairs(
    ancient_model,
    modern_model,
    ancient_corpus=None,
    modern_corpus=None,
    anc_uni=None, anc_cooc=None, anc_total=None,
    mod_uni=None, mod_cooc=None, mod_total=None,
    pairs=None,
    output_path=None,
    pmi_window=5,
):
    """
    Pair-level cosine similarity (within each model) and PPMI (within each corpus).
    Accepts pre-built co-occurrence tables to avoid recomputation.
    """
    if pairs is None:
        pairs = PAIRS

    pmi_available, anc_uni, anc_cooc, anc_total, mod_uni, mod_cooc, mod_total = \
        _ensure_cooc_tables(
            anc_uni, anc_cooc, anc_total,
            mod_uni, mod_cooc, mod_total,
            ancient_corpus, modern_corpus,
            pmi_window,
        )

    output_lines = []

    # -----------------------------------------------------------------------
    # Collect results
    # -----------------------------------------------------------------------
    results = []
    for word1, word2, label in pairs:
        a_sim, _ = cosine_sim(ancient_model, word1, word2)
        m_sim, _ = cosine_sim(modern_model,  word1, word2)
        a_pmi = pmi_score(word1, word2, anc_uni, anc_cooc, anc_total) if pmi_available else None
        m_pmi = pmi_score(word1, word2, mod_uni, mod_cooc, mod_total) if pmi_available else None
        results.append((label, word1, word2, a_sim, m_sim, a_pmi, m_pmi))

    # -----------------------------------------------------------------------
    # Cosine similarity table
    # -----------------------------------------------------------------------
    cos_table = Table(
        title="\nPair Cosine Similarity (within each model)",
        box=box.ROUNDED, show_lines=True,
        title_style="bold cyan", header_style="bold magenta",
    )
    cos_table.add_column("Pair",        style="bold white", justify="left",   min_width=32)
    cos_table.add_column("Tang",        style="yellow",     justify="center", min_width=10)
    cos_table.add_column("Modern",      style="green",      justify="center", min_width=10)
    cos_table.add_column("Δ(mod−tang)", style="cyan",       justify="center", min_width=10)
    cos_table.add_column("Direction",   justify="left",     min_width=24)

    output_lines.append("\nPAIR COSINE SIMILARITY\n" + "=" * 80 + "\n")
    for label, w1, w2, a_sim, m_sim, _, __ in results:
        a_str = f"{a_sim:.4f}" if a_sim is not None else "[red]OOV[/red]"
        m_str = f"{m_sim:.4f}" if m_sim is not None else "[red]OOV[/red]"
        if a_sim is not None and m_sim is not None:
            delta = m_sim - a_sim
            d_str = f"{delta:+.4f}"
            direction = (
                "[green]↑ closer in modern[/green]"  if delta >  0.01 else
                "[red]↓ further in modern[/red]"     if delta < -0.01 else
                "[yellow]≈ stable[/yellow]"
            )
            output_lines.append(f"  {label:<34} Tang={a_sim:.4f}  Modern={m_sim:.4f}  Δ={delta:+.4f}\n")
        else:
            d_str, direction = "N/A", ""
            output_lines.append(f"  {label:<34} OOV\n")
        cos_table.add_row(label, a_str, m_str, d_str, direction)

    console.print(cos_table)

    # -----------------------------------------------------------------------
    # PPMI table
    # -----------------------------------------------------------------------
    if pmi_available:
        pmi_table = Table(
            title=f"Pair PPMI (character-level, window={pmi_window})",
            box=box.ROUNDED, show_lines=True,
            title_style="bold cyan", header_style="bold magenta",
        )
        pmi_table.add_column("Pair",        style="bold white", justify="left",   min_width=32)
        pmi_table.add_column("Tang PPMI",   style="yellow",     justify="center", min_width=10)
        pmi_table.add_column("Modern PPMI", style="green",      justify="center", min_width=10)
        pmi_table.add_column("Δ(mod−tang)", style="cyan",       justify="center", min_width=10)
        pmi_table.add_column("Direction",   justify="left",     min_width=24)

        output_lines.append(f"\nPAIR PPMI (window={pmi_window})\n" + "=" * 80 + "\n")
        for label, w1, w2, _, __, a_pmi, m_pmi in results:
            a_str = f"{a_pmi:.4f}" if a_pmi is not None else "[red]no co-occ[/red]"
            m_str = f"{m_pmi:.4f}" if m_pmi is not None else "[red]no co-occ[/red]"
            if a_pmi is not None and m_pmi is not None:
                delta = m_pmi - a_pmi
                d_str = f"{delta:+.4f}"
                direction = (
                    "[green]↑ stronger in modern[/green]" if delta >  0.05 else
                    "[red]↓ weaker in modern[/red]"       if delta < -0.05 else
                    "[yellow]≈ stable[/yellow]"
                )
                output_lines.append(f"  {label:<34} Tang={a_pmi:.4f}  Modern={m_pmi:.4f}  Δ={delta:+.4f}\n")
            else:
                d_str, direction = "N/A", ""
                output_lines.append(f"  {label:<34} no co-occurrence\n")
            pmi_table.add_row(label, a_str, m_str, d_str, direction)

        console.print(pmi_table)

    # -----------------------------------------------------------------------
    # Write output file
    # -----------------------------------------------------------------------
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.writelines(output_lines)
        print(f"Saved: {output_path}")

    return results


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Loading saved models...")
    ancient_model = Word2Vec.load("model/tang_word2vec.model")
    modern_model  = Word2Vec.load("model/modern_word2vec.model")
    analyze_pairs(ancient_model, modern_model)