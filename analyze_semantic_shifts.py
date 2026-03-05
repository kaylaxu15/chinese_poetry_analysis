from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from rich.console import Console
from rich.table import Table
from rich import box
import numpy as np

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

def cosine_sim(model, word1, word2):
    """Return cosine similarity between two words in a model, or None if OOV."""
    if word1 not in model.wv or word2 not in model.wv:
        missing = [w for w in (word1, word2) if w not in model.wv]
        return None, missing
    v1 = model.wv[word1].reshape(1, -1)
    v2 = model.wv[word2].reshape(1, -1)
    return float(cosine_similarity(v1, v2)[0][0]), []


def analyze_pairs(ancient_model, modern_model, pairs=None, output_path=None):
    """
    Print a side-by-side comparison of cosine similarities for each word pair
    across ancient (Tang) and modern poetry models.

    Args:
        ancient_model: trained gensim Word2Vec model for Tang poetry
        modern_model:  trained gensim Word2Vec model for modern poetry
        pairs:         list of (word1, word2, label) tuples; defaults to PAIRS
    """
    if pairs is None:
        pairs = PAIRS

    # ---------------------------------------------------------------------------
    # Main results table
    # ---------------------------------------------------------------------------
    table = Table(
        title="\nSemantic Shift: Romanticized Relationships with Nature & Dreams",
        box=box.ROUNDED,
        show_lines=True,
        title_style="bold cyan",
        header_style="bold magenta",
    )

    table.add_column("Pair",              style="bold white",  justify="left",   min_width=32)
    table.add_column("Tang (ancient)",    style="yellow",      justify="center", min_width=14)
    table.add_column("Modern",            style="green",       justify="center", min_width=14)
    table.add_column("Δ (modern−tang)",   style="cyan",        justify="center", min_width=14)
    table.add_column("Direction",         justify="left",      min_width=20)

    results = []
    for word1, word2, label in pairs:
        ancient_sim, ancient_missing = cosine_sim(ancient_model, word1, word2)
        modern_sim,  modern_missing  = cosine_sim(modern_model,  word1, word2)

        ancient_str = f"{ancient_sim:.4f}" if ancient_sim is not None else f"[red]OOV: {ancient_missing}[/red]"
        modern_str  = f"{modern_sim:.4f}"  if modern_sim  is not None else f"[red]OOV: {modern_missing}[/red]"

        if ancient_sim is not None and modern_sim is not None:
            delta = modern_sim - ancient_sim
            delta_str = f"{delta:+.4f}"
            if delta > 0.01:
                direction = "[green]↑ closer in modern[/green]"
            elif delta < -0.01:
                direction = "[red]↓ further in modern[/red]"
            else:
                direction = "[yellow]≈ stable[/yellow]"
        else:
            delta_str = "N/A"
            direction = ""

        table.add_row(label, ancient_str, modern_str, delta_str, direction)
        results.append((label, ancient_sim, modern_sim))

    console.print(table)

    # ---------------------------------------------------------------------------
    # Interpretation summary table
    # ---------------------------------------------------------------------------
    summary = Table(
        title="Interpretation",
        box=box.SIMPLE_HEAVY,
        title_style="bold cyan",
        header_style="bold magenta",
        show_lines=False,
    )
    summary.add_column("Pair",    style="bold white", min_width=24)
    summary.add_column("Finding", min_width=52)

    for label, a_sim, m_sim in results:
        if a_sim is None or m_sim is None:
            continue
        delta = m_sim - a_sim
        pair_short = label.split("(")[0].strip()
        if delta > 0.05:
            finding = f"[green]Concepts MORE linked in modern poetry (+{delta:.3f})[/green]"
        elif delta < -0.05:
            finding = f"[red]Concepts LESS linked in modern poetry ({delta:.3f})[/red]"
        else:
            finding = f"[yellow]Relationship largely stable across periods ({delta:+.3f})[/yellow]"
        summary.add_row(pair_short, finding)

    console.print(summary)

    # ---------------------------------------------------------------------------
    # Write plain-text version to file
    # ---------------------------------------------------------------------------
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("Semantic Shift: Romanticized Relationships with Nature & Dreams\n")
            f.write("=" * 70 + "\n")
            f.write(f"  {'Pair':<34} {'Tang':>10}   {'Modern':>10}   {'Δ':>10}   Direction\n")
            f.write("-" * 70 + "\n")
            for label, a_sim, m_sim in results:
                if a_sim is not None and m_sim is not None:
                    delta = m_sim - a_sim
                    direction = "↑ closer in modern" if delta > 0.01 else ("↓ further in modern" if delta < -0.01 else "≈ stable")
                    f.write(f"  {label:<34} {a_sim:>10.4f}   {m_sim:>10.4f}   {delta:>+10.4f}   {direction}\n")
                else:
                    f.write(f"  {label:<34} {'OOV':>10}   {'OOV':>10}   {'N/A':>10}\n")
            f.write("\nInterpretation:\n")
            for label, a_sim, m_sim in results:
                if a_sim is None or m_sim is None:
                    continue
                delta = m_sim - a_sim
                pair_short = label.split("(")[0].strip()
                if delta > 0.05:
                    f.write(f"  • {pair_short}: concepts MORE linked in modern poetry (+{delta:.3f})\n")
                elif delta < -0.05:
                    f.write(f"  • {pair_short}: concepts LESS linked in modern poetry ({delta:.3f})\n")
                else:
                    f.write(f"  • {pair_short}: relationship largely stable across periods ({delta:+.3f})\n")
        print(f"Saved: {output_path}")

    return results


# ---------------------------------------------------------------------------
# Standalone entry point — loads saved models and runs analysis
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Loading saved models...")
    ancient_model = Word2Vec.load("tang_word2vec.model")
    modern_model  = Word2Vec.load("modern_word2vec.model")
    analyze_pairs(ancient_model, modern_model)
