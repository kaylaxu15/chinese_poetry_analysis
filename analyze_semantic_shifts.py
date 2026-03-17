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
# Word pairs of interest: romanticized relationships with nature and dreams
# ---------------------------------------------------------------------------
PAIRS = [
    ("人", "月",  "person–moon   (nature/romance)"),
    ("人", "梦",  "person–dream  (dream)"),
    ("人", "花",  "person–flower (nature/beauty)"),
    ("月", "梦",  "moon–dream    (romanticized dream)"),
]


# ---------------------------------------------------------------------------
# Character-level tokenization helpers
# ---------------------------------------------------------------------------

def char_tokenize_corpus(corpus):
    """
    Re-tokenize a corpus (list of token lists) into individual characters.
    Each poem becomes a flat list of single characters, stripping whitespace.

    Args:
        corpus: list of lists of strings (word-tokenized poems)
    Returns:
        list of lists of single-character strings
    """
    char_corpus = []
    for tokens in corpus:
        chars = [ch for token in tokens for ch in token if ch.strip()]
        if chars:
            char_corpus.append(chars)
    return char_corpus


# ---------------------------------------------------------------------------
# PMI helpers
# ---------------------------------------------------------------------------

def build_cooccurrence(char_corpus, window=5):
    """
    Build unigram and co-occurrence counts from a character-tokenized corpus.

    Args:
        char_corpus: list of lists of single characters
        window:      symmetric context window size
    Returns:
        unigram_counts: dict {char: count}
        cooc_counts:    dict {(char1, char2): count}  (unordered pairs, char1 < char2)
        total_tokens:   int
    """
    unigram_counts = defaultdict(int)
    cooc_counts    = defaultdict(int)
    total_tokens   = 0

    for doc in char_corpus:
        total_tokens += len(doc)
        for i, target in enumerate(doc):
            unigram_counts[target] += 1
            start = max(0, i - window)
            end   = min(len(doc), i + window + 1)
            for j in range(start, end):
                if j == i:
                    continue
                context = doc[j]
                pair = (min(target, context), max(target, context))
                cooc_counts[pair] += 1

    return unigram_counts, cooc_counts, total_tokens


def pmi_score(word1, word2, unigram_counts, cooc_counts, total_tokens, positive=True):
    """
    Compute (Positive) Pointwise Mutual Information for a word pair.

    PMI(w1, w2) = log2[ P(w1,w2) / (P(w1) * P(w2)) ]
    PPMI clamps negative values to 0.

    Args:
        word1, word2:    characters to score
        unigram_counts:  dict from build_cooccurrence()
        cooc_counts:     dict from build_cooccurrence()
        total_tokens:    total token count from build_cooccurrence()
        positive:        if True, return PPMI (clamp negatives to 0)
    Returns:
        float score, or None if either word is unseen or never co-occurs
    """
    pair = (min(word1, word2), max(word1, word2))
    cooc = cooc_counts.get(pair, 0)
    if cooc == 0:
        return None

    # Number of (target, context) events = 2 * cooc_counts (symmetric window)
    total_cooc = sum(cooc_counts.values())

    p_w1   = unigram_counts[word1] / total_tokens
    p_w2   = unigram_counts[word2] / total_tokens
    p_w1w2 = cooc / total_cooc

    if p_w1 == 0 or p_w2 == 0:
        return None

    raw_pmi = math.log2(p_w1w2 / (p_w1 * p_w2))
    return max(0.0, raw_pmi) if positive else raw_pmi


# ---------------------------------------------------------------------------
# Cosine similarity helper (unchanged)
# ---------------------------------------------------------------------------

def cosine_sim(model, word1, word2):
    """Return cosine similarity between two words in a model, or None if OOV."""
    if word1 not in model.wv or word2 not in model.wv:
        missing = [w for w in (word1, word2) if w not in model.wv]
        return None, missing
    v1 = model.wv[word1].reshape(1, -1)
    v2 = model.wv[word2].reshape(1, -1)
    return float(cosine_similarity(v1, v2)[0][0]), []


# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------

def analyze_pairs(
    ancient_model,
    modern_model,
    ancient_corpus=None,
    modern_corpus=None,
    pairs=None,
    output_path=None,
    pmi_window=5,
):
    """
    Print a side-by-side comparison of cosine similarities AND PPMI scores
    for each word pair across ancient (Tang) and modern poetry models.

    Cosine similarity uses the Word2Vec embeddings as before.
    PPMI is computed from character-level co-occurrence statistics derived
    from the raw corpora (re-tokenized to single characters internally).

    Args:
        ancient_model:   trained gensim Word2Vec model for Tang poetry
        modern_model:    trained gensim Word2Vec model for modern poetry
        ancient_corpus:  original word-tokenized corpus (list of token lists)
                         used to build char-level PMI; PMI skipped if None
        modern_corpus:   same for modern poetry
        pairs:           list of (word1, word2, label); defaults to PAIRS
        output_path:     path to write plain-text results
        pmi_window:      context window for co-occurrence counts
    """
    if pairs is None:
        pairs = PAIRS

    # -----------------------------------------------------------------------
    # Build character-level co-occurrence tables if corpora supplied
    # -----------------------------------------------------------------------
    pmi_available = ancient_corpus is not None and modern_corpus is not None

    if pmi_available:
        ancient_chars = char_tokenize_corpus(ancient_corpus)
        modern_chars  = char_tokenize_corpus(modern_corpus)

        anc_uni, anc_cooc, anc_total = build_cooccurrence(ancient_chars, window=pmi_window)
        mod_uni, mod_cooc, mod_total = build_cooccurrence(modern_chars,  window=pmi_window)

    # -----------------------------------------------------------------------
    # Collect results
    # -----------------------------------------------------------------------
    results = []
    for word1, word2, label in pairs:
        ancient_sim, ancient_missing = cosine_sim(ancient_model, word1, word2)
        modern_sim,  modern_missing  = cosine_sim(modern_model,  word1, word2)

        anc_pmi = mod_pmi = None
        if pmi_available:
            anc_pmi = pmi_score(word1, word2, anc_uni, anc_cooc, anc_total)
            mod_pmi = pmi_score(word1, word2, mod_uni, mod_cooc, mod_total)

        results.append((label, word1, word2, ancient_sim, modern_sim, anc_pmi, mod_pmi))

    # -----------------------------------------------------------------------
    # Cosine similarity table
    # -----------------------------------------------------------------------
    cos_table = Table(
        title="\nSemantic Shift: Romanticized Relationships — Cosine Similarity",
        box=box.ROUNDED,
        show_lines=True,
        title_style="bold cyan",
        header_style="bold magenta",
    )
    cos_table.add_column("Pair",            style="bold white", justify="left",   min_width=32)
    cos_table.add_column("Tang",  style="yellow",     justify="center", min_width=14)
    cos_table.add_column("Modern",          style="green",      justify="center", min_width=14)
    cos_table.add_column("Δ (mod−tang)",    style="cyan",       justify="center", min_width=14)
    cos_table.add_column("Direction",       justify="left",     min_width=20)

    for label, w1, w2, a_sim, m_sim, _, __ in results:
        a_str = f"{a_sim:.4f}" if a_sim is not None else "[red]OOV[/red]"
        m_str = f"{m_sim:.4f}" if m_sim is not None else "[red]OOV[/red]"

        if a_sim is not None and m_sim is not None:
            delta = m_sim - a_sim
            d_str = f"{delta:+.4f}"
            if delta > 0.01:
                direction = "[green]↑Strong association in modern[/green]"
            elif delta < -0.01:
                direction = "[red]Weaker association in modern[/red]"
            else:
                direction = "[yellow]≈ stable[/yellow]"
        else:
            d_str, direction = "N/A", ""

        cos_table.add_row(label, a_str, m_str, d_str, direction)

    console.print(cos_table)

    # -----------------------------------------------------------------------
    # PPMI table (only if corpora were supplied)
    # -----------------------------------------------------------------------
    if pmi_available:
        pmi_table = Table(
            title="Semantic Shift: Romanticized Relationships — PPMI (char-level)",
            box=box.ROUNDED,
            show_lines=True,
            title_style="bold cyan",
            header_style="bold magenta",
        )
        pmi_table.add_column("Pair",           style="bold white", justify="left",   min_width=32)
        pmi_table.add_column("Tang PPMI",       style="yellow",     justify="center", min_width=14)
        pmi_table.add_column("Modern PPMI",     style="green",      justify="center", min_width=14)
        pmi_table.add_column("Δ (mod−tang)",    style="cyan",       justify="center", min_width=14)
        pmi_table.add_column("Direction",       justify="left",     min_width=20)

        for label, w1, w2, _, __, a_pmi, m_pmi in results:
            a_str = f"{a_pmi:.4f}" if a_pmi is not None else "[red]no co-occ[/red]"
            m_str = f"{m_pmi:.4f}" if m_pmi is not None else "[red]no co-occ[/red]"

            if a_pmi is not None and m_pmi is not None:
                delta = m_pmi - a_pmi
                d_str = f"{delta:+.4f}"
                if delta > 0.05:
                    direction = "[green]↑ Stronger association in modern[/green]"
                elif delta < -0.05:
                    direction = "[red]↓ Weaker association in modern[/red]"
                else:
                    direction = "[yellow]≈ similar[/yellow]"
            else:
                d_str, direction = "N/A", ""

            pmi_table.add_row(label, a_str, m_str, d_str, direction)

        console.print(pmi_table)

    # -----------------------------------------------------------------------
    # Interpretation summary
    # -----------------------------------------------------------------------
    summary = Table(
        title="Interpretation",
        box=box.SIMPLE_HEAVY,
        title_style="bold cyan",
        header_style="bold magenta",
        show_lines=True,
    )
    summary.add_column("Pair",         style="bold white", justify="left",   min_width=20)
    summary.add_column("ΔCosine (Modern - Tang)",     style="cyan",       justify="center", min_width=10)
    summary.add_column("Cosine finding",                   justify="left",   min_width=30)
    if pmi_available:
        summary.add_column("Tang PPMI",  style="yellow",   justify="center", min_width=10)
        summary.add_column("Modern PPMI",   style="green",    justify="center", min_width=10)
        summary.add_column("ΔPPMI (Modern - Tang)",     style="cyan",     justify="center", min_width=10)
        summary.add_column("PPMI finding",                 justify="left",   min_width=30)

    for label, w1, w2, a_sim, m_sim, a_pmi, m_pmi in results:
        pair_short = label.split("(")[0].strip()

        # cosine finding
        if a_sim is not None and m_sim is not None:
            cos_d = m_sim - a_sim
            cos_d_str = f"{cos_d:+.4f}"
            if cos_d > 0.05:
                cos_finding = f"[green]Stronger relationship in modern[/green]"
            elif cos_d < -0.05:
                cos_finding = f"[red]Weaker relationship in modern[/red]"
            else:
                cos_finding = f"[yellow]≈ similar[/yellow]"
        else:
            cos_d_str   = "N/A"
            cos_finding = "[red]OOV[/red]"

        if pmi_available:
            if a_pmi is not None and m_pmi is not None:
                pmi_d = m_pmi - a_pmi
                a_pmi_str = f"{a_pmi:.4f}"
                m_pmi_str = f"{m_pmi:.4f}"
                pmi_d_str = f"{pmi_d:+.4f}"
                if pmi_d > 0.05:
                    pmi_finding = "[green]stronger association in modern[/green]"
                elif pmi_d < -0.05:
                    pmi_finding = "[red]weaker association in modern[/red]"
                else:
                    pmi_finding = "[yellow]≈ similar[/yellow]"
            else:
                a_pmi_str = m_pmi_str = pmi_d_str = "—"
                pmi_finding = "[red]no co-occurrence[/red]"
            summary.add_row(pair_short, cos_d_str, cos_finding,
                            a_pmi_str, m_pmi_str, pmi_d_str, pmi_finding)
        else:
            summary.add_row(pair_short, cos_d_str, cos_finding)

    console.print(summary)

    # -----------------------------------------------------------------------
    # Write plain-text file
    # -----------------------------------------------------------------------
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("Semantic Shift: Romanticized Relationships with Nature & Dreams\n")
            f.write("=" * 80 + "\n\n")

            # Cosine section
            f.write("COSINE SIMILARITY (Word2Vec embeddings)\n")
            f.write("-" * 80 + "\n")
            f.write(f"  {'Pair':<34} {'Tang':>10}   {'Modern':>10}   {'Δ':>10}   Direction\n")
            f.write("-" * 80 + "\n")
            for label, w1, w2, a_sim, m_sim, _, __ in results:
                if a_sim is not None and m_sim is not None:
                    delta = m_sim - a_sim
                    direction = "↑ closer in modern" if delta > 0.01 else ("↓ further in modern" if delta < -0.01 else "≈ stable")
                    f.write(f"  {label:<34} {a_sim:>10.4f}   {m_sim:>10.4f}   {delta:>+10.4f}   {direction}\n")
                else:
                    f.write(f"  {label:<34} {'OOV':>10}   {'OOV':>10}   {'N/A':>10}\n")

            # PMI section
            if pmi_available:
                f.write(f"\n\nPPMI — POINTWISE MUTUAL INFORMATION (character-level, window={pmi_window})\n")
                f.write("-" * 80 + "\n")
                f.write(f"  {'Pair':<34} {'Tang PPMI':>10}   {'Mod PPMI':>10}   {'Δ':>10}   Direction\n")
                f.write("-" * 80 + "\n")
                for label, w1, w2, _, __, a_pmi, m_pmi in results:
                    if a_pmi is not None and m_pmi is not None:
                        delta = m_pmi - a_pmi
                        direction = "↑ stronger in modern" if delta > 0.1 else ("↓ weaker in modern" if delta < -0.1 else "≈ stable")
                        f.write(f"  {label:<34} {a_pmi:>10.4f}   {m_pmi:>10.4f}   {delta:>+10.4f}   {direction}\n")
                    else:
                        f.write(f"  {label:<34} {'—':>10}   {'—':>10}   {'N/A':>10}   no co-occurrence\n")

            # Interpretation
            f.write("\n\nInterpretation:\n")
            for label, w1, w2, a_sim, m_sim, a_pmi, m_pmi in results:
                pair_short = label.split("(")[0].strip()
                f.write(f"\n  {pair_short}:\n")
                if a_sim is not None and m_sim is not None:
                    d = m_sim - a_sim
                    if d > 0.05:
                        f.write(f"    Cosine: concepts MORE linked in modern poetry (+{d:.3f})\n")
                    elif d < -0.05:
                        f.write(f"    Cosine: concepts LESS linked in modern poetry ({d:.3f})\n")
                    else:
                        f.write(f"    Cosine: relationship largely stable ({d:+.3f})\n")
                if pmi_available and a_pmi is not None and m_pmi is not None:
                    d = m_pmi - a_pmi
                    if d > 0.5:
                        f.write(f"    PPMI:   association STRONGER in modern poetry (+{d:.3f})\n")
                    elif d < -0.5:
                        f.write(f"    PPMI:   association WEAKER in modern poetry ({d:.3f})\n")
                    else:
                        f.write(f"    PPMI:   association largely stable ({d:+.3f})\n")

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