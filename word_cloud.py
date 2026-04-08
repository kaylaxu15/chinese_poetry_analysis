import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sentence_transformers import SentenceTransformer
from collections import Counter
from disappearing_words import top_collocates

cjk_font = fm.FontProperties(fname="/System/Library/Fonts/STHeiti Light.ttc")
fm.fontManager.addfont("/System/Library/Fonts/STHeiti Light.ttc")
FONT_NAME = cjk_font.get_name()

embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

def cosine_sim(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

def plot_radial_collocates(center_char, collocates, title, filename, color):
    """
    collocates: list of (char, bigram_freq) tuples
    Positions each collocate at angle=evenly spaced, radius=1-cosine_sim (closer=more similar)
    Font size scales with bigram frequency.
    """
    if not collocates:
        return

    center_vec = embedding_model.encode(center_char, normalize_embeddings=True)

    # Compute cosine similarity and distance for each collocate
    words, freqs = zip(*collocates)
    vecs = embedding_model.encode(list(words), normalize_embeddings=True)
    sims = np.array([cosine_sim(center_vec, v) for v in vecs])

    # Radius: low similarity → far from center. Scale to [0.2, 0.9] of plot radius.
    # Invert: distance = 1 - sim, then remap to [0.2, 0.9]
    raw_dist = 1.0 - sims
    d_min, d_max = raw_dist.min(), raw_dist.max()
    if d_max - d_min < 1e-6:
        norm_dist = np.full_like(raw_dist, 0.5)
    else:
        norm_dist = 0.2 + 0.7 * (raw_dist - d_min) / (d_max - d_min)

    # Evenly spread angles, offset by a little so nothing sits at 0°
    n = len(words)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + np.pi / n

    # Font size: scale with frequency, range [10, 22]
    freqs_arr = np.array(freqs, dtype=float)
    f_min, f_max = freqs_arr.min(), freqs_arr.max()
    if f_max - f_min < 1e-6:
        font_sizes = np.full(n, 14.0)
    else:
        font_sizes = 10 + 12 * (freqs_arr - f_min) / (f_max - f_min)

    # Alpha: scale with similarity [0.45, 1.0]
    alphas = 0.45 + 0.55 * (sims - sims.min()) / (sims.max() - sims.min() + 1e-9)

    fig, ax = plt.subplots(figsize=(7, 7), facecolor='white')
    ax.set_aspect('equal')
    ax.axis('off')

    RADIUS = 3.2  # coordinate radius of the plot area

    # Draw faint reference rings
    for r_frac in [0.33, 0.66, 1.0]:
        ring = plt.Circle((0, 0), RADIUS * r_frac, fill=False,
                           color='#cccccc', linewidth=0.5, linestyle='--', zorder=0)
        ax.add_patch(ring)

    # Draw center character
    center_circle = plt.Circle((0, 0), 0.38, color=color, zorder=3, alpha=0.15)
    ax.add_patch(center_circle)
    ax.text(0, 0, center_char, ha='center', va='center',
            fontproperties=cjk_font, fontsize=28, fontweight='bold',
            color=color, zorder=4)

    # Draw collocates
    for i, (word, freq) in enumerate(zip(words, freqs)):
        x = norm_dist[i] * RADIUS * np.cos(angles[i])
        y = norm_dist[i] * RADIUS * np.sin(angles[i])

        # Thin leader line from center edge to word
        line_start = 0.42  # just outside center circle
        lx = line_start * np.cos(angles[i])
        ly = line_start * np.sin(angles[i])
        ax.plot([lx, x * 0.88], [ly, y * 0.88],
                color='#cccccc', linewidth=0.5, zorder=1)

        ax.text(x, y, word,
                ha='center', va='center',
                fontproperties=cjk_font,
                fontsize=font_sizes[i],
                color=color,
                alpha=float(alphas[i]),
                zorder=2)

    # Ring annotations
    ax.text(RADIUS * 0.33 + 0.08, 0.08, 'high sim',
            fontsize=8, color='#aaaaaa', va='bottom')
    ax.text(RADIUS * 1.0 + 0.08, 0.08, 'low sim',
            fontsize=8, color='#aaaaaa', va='bottom')

    ax.set_xlim(-RADIUS - 0.6, RADIUS + 1.2)
    ax.set_ylim(-RADIUS - 0.6, RADIUS + 0.6)

    ax.set_title(title, fontproperties=cjk_font, fontsize=13, pad=12, color='#333333')

    # Subtitle
    fig.text(0.5, 0.02,
             'Distance = semantic similarity   |   Font size = bigram frequency',
             ha='center', fontsize=8, color='#999999')

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {filename}")


def plot_top10_radial(results, classical_bigrams, modern_bigrams):
    configs = [
        (results["top20_classical"][:10], classical_bigrams, '#7F77DD', 'classical'),
        (results["top20_modern"][:10],    modern_bigrams,    '#1D9E75', 'modern'),
    ]

    for top10, bigrams, color, label in configs:
        for rank, (ch, freq) in enumerate(top10):
            collocates = top_collocates(ch, bigrams, n=12)
            if not collocates:
                continue
            title = f'"{ch}"  (freq {freq}) — top collocates in {label} corpus'
            filename = f'radial_{label}_{rank+1:02d}_{ch}.png'
            plot_radial_collocates(ch, collocates, title, filename, color)