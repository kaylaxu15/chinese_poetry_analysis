import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

cjk_font = fm.FontProperties(fname="/System/Library/Fonts/STHeiti Light.ttc")
fm.fontManager.addfont("/System/Library/Fonts/STHeiti Light.ttc")

qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
qwen_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", torch_dtype=torch.float16, device_map="auto")

def get_translations(words):
    """Fetch single-character English translations using Qwen."""
    prompt = (
        f"Translate each of these Chinese characters individually into a single English word: "
        f"{', '.join(words)}. "
        f"Respond with only a JSON object mapping each character to one English word, "
        f"e.g. {{\"风\": \"wind\", \"水\": \"water\"}}. "
        f"One word per character, no phrases, no explanations, no other text."
    )
    messages = [{"role": "user", "content": prompt}]
    text = qwen_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = qwen_tokenizer(text, return_tensors="pt").to(qwen_model.device)
    with torch.no_grad():
        outputs = qwen_model.generate(**inputs, max_new_tokens=300, do_sample=False)
    response = qwen_tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return {w: "" for w in words}

def plot_radial_equivalents(center_char, equivalents, title, filename, color):
    """
    equivalents: list of (word, cosine_sim_score) tuples from find_modern_equivalent()
    """
    if not equivalents:
        return

    words, sims = zip(*equivalents)
    sims = np.array(sims, dtype=float)

    # Fetch translations for all words in one call
    translations = get_translations(list(words))

    # Distance: map cosine sim [0, 1] directly to radius [0.9, 0.2]
    norm_dist = 0.9 - 0.7 * sims

    # Evenly spread angles
    n = len(words)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + np.pi / n

    # Font size: higher similarity → larger, range [10, 24]
    font_sizes = 10 + 14 * (sims - sims.min()) / (sims.max() - sims.min() + 1e-9)

    fig, ax = plt.subplots(figsize=(9, 9), facecolor='white')
    ax.set_aspect('equal')
    ax.axis('off')

    RADIUS = 3.2

    # Faint reference rings
    for r_frac in [0.33, 0.66, 1.0]:
        ring = plt.Circle((0, 0), RADIUS * r_frac, fill=False,
                           color='#cccccc', linewidth=0.5, linestyle='--', zorder=0)
        ax.add_patch(ring)

    # Center character
    center_circle = plt.Circle((0, 0), 0.38, color=color, zorder=3, alpha=0.15)
    ax.add_patch(center_circle)
    ax.text(0, 0, center_char, ha='center', va='center',
            fontproperties=cjk_font, fontsize=28, fontweight='bold',
            color=color, zorder=4)

    # Equivalents
    for i, word in enumerate(words):
        x = norm_dist[i] * RADIUS * np.cos(angles[i])
        y = norm_dist[i] * RADIUS * np.sin(angles[i])

        lx = 0.42 * np.cos(angles[i])
        ly = 0.42 * np.sin(angles[i])
        ax.plot([lx, x * 0.88], [ly, y * 0.88],
                color='#cccccc', linewidth=0.5, zorder=1)

        translation = translations.get(word, "")
        label = f"{word}: {translation} ({sims[i]:.2f})" if translation else f"{word} ({sims[i]:.2f})"

        # CJK character in larger font
        ax.text(x, y + 0.15, word,
                ha='center', va='center',
                fontproperties=cjk_font,
                fontsize=float(font_sizes[i]),
                color=color,
                zorder=2)
        # Translation and score in smaller font below
        ax.text(x, y - 0.22, f"{translation} ({sims[i]:.2f})" if translation else f"({sims[i]:.2f})",
                ha='center', va='center',
                fontsize=float(font_sizes[i]) * 0.6,
                color=color,
                zorder=2)

    ax.set_xlim(-RADIUS - 0.8, RADIUS + 0.8)
    ax.set_ylim(-RADIUS - 0.8, RADIUS + 0.8)
    ax.set_title(title, fontproperties=cjk_font, fontsize=13, pad=12, color='#333333')
    fig.text(0.5, 0.02,
             'Distance and size = semantic similarity   |   Closer and larger = more similar',
             ha='center', fontsize=8, color='#999999')

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {filename}")


def plot_top15_equivalents(truly_disappeared, find_modern_equivalent_fn,
                           model_classical, modern_aligned):
    for rank, (ch, freq) in enumerate(truly_disappeared[:15]):
        equivalents = find_modern_equivalent_fn(ch, model_classical, modern_aligned, topn=10)
        if not equivalents:
            continue
        title = f'"{ch}"  (freq {freq}) — modern equivalents'
        filename = f'radial_equiv_{rank+1:02d}_{ch}.png'
        plot_radial_equivalents(ch, equivalents, title, filename, color='#7F77DD')