import re
import itertools
import json
from pathlib import Path
import numpy as np
from matplotlib import rcParams
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from turftopic import KeyNMF
from count_ancient_tokens import classical_poems
from count_modern_tokens import modern_poems
from sklearn.feature_extraction.text import CountVectorizer
import joblib
from topic_plotting import plot_topics_3d

rcParams['font.sans-serif'] = ['Arial Unicode MS']
rcParams['axes.unicode_minus'] = False

# -----------------------------
# Text cleaning
# -----------------------------
def clean_text(tokens):
    text = "".join(tokens)
    text = re.sub(r"[^\u4e00-\u9fff]", "", text)
    return text

classical_docs = [clean_text(tokens) for tokens in classical_poems]
modern_docs    = [clean_text(tokens) for tokens in modern_poems]

embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# -----------------------------
# Model paths
# -----------------------------
model_c_file           = Path("model_c.joblib")
model_m_file           = Path("model_m.joblib")
poem_embeddings_c_file = Path("poem_embeddings_c.npy")
poem_embeddings_m_file = Path("poem_embeddings_m.npy")

# -----------------------------
# Load or create+fit models
# -----------------------------
if model_c_file.exists() and model_m_file.exists():
    print("Loading saved models...")
    model_c = joblib.load(model_c_file)
    model_m = joblib.load(model_m_file)

    if not hasattr(model_c.vectorizer, 'vocabulary_'):
        print("Classical model vectorizer not fitted — refitting...")
        model_c.fit(classical_docs)
        joblib.dump(model_c, model_c_file)

    if not hasattr(model_m.vectorizer, 'vocabulary_'):
        print("Modern model vectorizer not fitted — refitting...")
        model_m.fit(modern_docs)
        joblib.dump(model_m, model_m_file)
else:
    print("No saved models found — creating and fitting new models...")

    model_c = KeyNMF(n_components=15, encoder=embedding_model, top_n=15, random_state=42,
                     vectorizer=CountVectorizer(analyzer="char", ngram_range=(1, 2), min_df=1))
    
    
    model_m = KeyNMF(n_components=15, encoder=embedding_model, top_n=15, random_state=42,
                     vectorizer=CountVectorizer(analyzer="char", ngram_range=(1, 2), min_df=1))
    model_c.fit(classical_docs)
    model_m.fit(modern_docs)
    joblib.dump(model_c, model_c_file)
    joblib.dump(model_m, model_m_file)

# -----------------------------
# Poem embeddings (cached to disk)
# -----------------------------
if poem_embeddings_c_file.exists():
    print("Loading classical poem embeddings...")
    poem_embeddings_c = np.load(poem_embeddings_c_file)
else:
    print("Encoding classical poems...")
    poem_embeddings_c = embedding_model.encode(
        classical_docs, normalize_embeddings=True, show_progress_bar=True
    )
    np.save(poem_embeddings_c_file, poem_embeddings_c)

if poem_embeddings_m_file.exists():
    print("Loading modern poem embeddings...")
    poem_embeddings_m = np.load(poem_embeddings_m_file)
else:
    print("Encoding modern poems...")
    poem_embeddings_m = embedding_model.encode(
        modern_docs, normalize_embeddings=True, show_progress_bar=True
    )
    np.save(poem_embeddings_m_file, poem_embeddings_m)

# -----------------------------
# Topic parsing (for keyword display only)
# -----------------------------
def parse_topic(topic, n=10):
    if isinstance(topic, dict):
        items = sorted(topic.items(), key=lambda x: x[1], reverse=True)[:n]
    elif isinstance(topic, list) and len(topic) > 0 and isinstance(topic[0], tuple):
        items = topic[:n]
    elif isinstance(topic, list) and len(topic) > 0 and isinstance(topic[0], str):
        items = [(w, 1.0) for w in topic[:n]]
    elif isinstance(topic, tuple) and len(topic) == 2:
        words, scores = topic
        if isinstance(scores, (int, float)):
            scores = [1.0] * len(words)
        elif isinstance(scores, (list, np.ndarray)):
            scores = list(scores)
        else:
            scores = [float(scores)] * len(words)
        items = list(zip(words[:n], scores[:n]))
    else:
        raise ValueError(f"Unexpected topic format: {type(topic)}")

    words = [w for w, _ in items]
    scores = np.array([s for _, s in items], dtype=float)
    if scores.sum() == 0:
        scores = np.ones_like(scores) / len(scores)
    else:
        scores = scores / scores.sum()
    return words, scores

def get_top_keywords(model, topic_idx, n=10):
    topics = model.get_topics()
    _, word_score_list = topics[topic_idx]
    words, _ = parse_topic(word_score_list, n)
    return words

# -----------------------------
# Document-topic matrices
# -----------------------------
W_classical = model_c.transform(classical_docs)
W_modern    = model_m.transform(modern_docs)

# -----------------------------
# Topic vectors: weighted average of poem embeddings
# -----------------------------
def compute_topic_vectors(W, poem_embeddings):
    topic_vecs = []
    for i in range(W.shape[1]):
        weights = W[:, i]
        weights = weights / (weights.sum() + 1e-9)
        vec = (weights[:, np.newaxis] * poem_embeddings).sum(axis=0)
        vec = vec / (np.linalg.norm(vec) + 1e-9)
        topic_vecs.append(vec.reshape(1, -1))
    return topic_vecs

print("Computing classical topic vectors from poem embeddings...")
c_vecs_all = compute_topic_vectors(W_classical, poem_embeddings_c)

print("Computing modern topic vectors from poem embeddings...")
m_vecs_all = compute_topic_vectors(W_modern, poem_embeddings_m)

# Keywords for display only
c_keywords_all = [get_top_keywords(model_c, i) for i in range(model_c.n_components)]
m_keywords_all = [get_top_keywords(model_m, i) for i in range(model_m.n_components)]

# -----------------------------
# Print top correspondences
# -----------------------------
def print_topics_with_full_weights(top_n_correspondences=5):
    similarities = []
    for i in range(model_c.n_components):
        for j in range(model_m.n_components):
            score = cosine_similarity(c_vecs_all[i], m_vecs_all[j])[0][0]
            c_weight = W_classical[:, i].mean()
            m_weight = W_modern[:, j].mean()
            similarities.append((score, i, j, c_weight, m_weight))

    similarities.sort(key=lambda x: x[0], reverse=True)
    top_correspondences = similarities[:top_n_correspondences]

    print(f"=== Top {top_n_correspondences} Classical→Modern Topic Correspondences ===\n")
    for rank, (score, i, j, c_weight, m_weight) in enumerate(top_correspondences):
        print(f"#{rank+1} Cosine: {score:.4f} | Classical {i+1} (w={c_weight:.4f}) → Modern {j+1} (w={m_weight:.4f})")
        print(f"  Classical keywords: {' '.join(c_keywords_all[i])}")
        print(f"  Modern keywords:    {' '.join(m_keywords_all[j])}\n")

    topic_weights_c = sorted(
        [(i, W_classical[:, i].mean()) for i in range(model_c.n_components)],
        key=lambda x: x[1], reverse=True
    )
    print("=== Classical Topics (sorted by weight) ===\n")
    for rank, (i, w) in enumerate(topic_weights_c):
        print(f"Classical Topic {rank+1} (weight {w:.4f})")
        print(f"  Top keywords: {' '.join(c_keywords_all[i])}\n")

    topic_weights_m = sorted(
        [(i, W_modern[:, i].mean()) for i in range(model_m.n_components)],
        key=lambda x: x[1], reverse=True
    )
    print("=== Modern Topics (sorted by weight) ===\n")
    for rank, (i, w) in enumerate(topic_weights_m):
        print(f"Modern Topic {rank+1} (weight {w:.4f})")
        print(f"  Top keywords: {' '.join(m_keywords_all[i])}\n")

print_topics_with_full_weights()

c_weights = np.array([W_classical[:, i].mean() for i in range(model_c.n_components)])
m_weights = np.array([W_modern[:, j].mean() for j in range(model_m.n_components)])

plot_topics_3d(c_vecs_all, m_vecs_all, c_keywords_all, m_keywords_all,
               c_weights, m_weights)