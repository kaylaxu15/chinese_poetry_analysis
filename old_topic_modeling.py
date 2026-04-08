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
from functools import lru_cache
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
model_c_file   = Path("model_c.joblib")
model_m_file   = Path("model_m.joblib")
keywords_c_file = Path("keywords_c.jsonl")
keywords_m_file = Path("keywords_m.jsonl")

# -----------------------------
# Load or create+fit models (single clean block)
# -----------------------------
if model_c_file.exists() and model_m_file.exists():
    print("Loading saved models...")
    model_c = joblib.load(model_c_file)
    model_m = joblib.load(model_m_file)

    # If saved models were never fitted (e.g. saved before .fit() was called), refit them
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
# Embedding helpers
# -----------------------------
@lru_cache(maxsize=None)
def embed_word(word):
    return embedding_model.encode(word, normalize_embeddings=True)

def embed_keywords(words):
    return np.array([embed_word(w) for w in words])

# -----------------------------
# Batched processing
# -----------------------------
def batched(iterable, n):
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(itertools.islice(it, n)):
        yield batch

def extract_and_save_keywords(model, docs, filename):
    with Path(filename).open("w", encoding="utf-8") as f:
        for batch in batched(docs, 200):
            batch_keywords = model.extract_keywords(list(batch))
            for kw in batch_keywords:
                f.write(json.dumps(kw, ensure_ascii=False) + "\n")

def stream_keywords(filename):
    with Path(filename).open(encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line.strip())
            except json.JSONDecodeError:
                continue  # skip bad lines

def fit_from_keywords(model, filename, epochs=3):
    for epoch in range(epochs):
        for batch in batched(stream_keywords(filename), 200):
            model.partial_fit(keywords=list(batch))

# -----------------------------
# Extract keywords if not already saved
# -----------------------------
if not keywords_c_file.exists():
    print("Extracting classical keywords...")
    extract_and_save_keywords(model_c, classical_docs, keywords_c_file)
else:
    print("Classical keywords file exists. Skipping extraction.")

if not keywords_m_file.exists():
    print("Extracting modern keywords...")
    extract_and_save_keywords(model_m, modern_docs, keywords_m_file)
else:
    print("Modern keywords file exists. Skipping extraction.")

# -----------------------------
# Topic parsing
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
    _, word_score_list = topics[topic_idx]   # unpack (idx, [(word, score), ...])
    words, _ = parse_topic(word_score_list, n)
    return words

def get_topic_embedding(model, topic_idx, n=10):
    topics = model.get_topics()
    _, word_score_list = topics[topic_idx]   # unpack (idx, [(word, score), ...])
    keywords, scores = parse_topic(word_score_list, n)
    return keywords, scores

# -----------------------------
# Document-topic matrices
# -----------------------------
W_classical = model_c.transform(classical_docs)
W_modern    = model_m.transform(modern_docs)

# -----------------------------
# Precompute topic embeddings
# -----------------------------
c_keywords_all, c_vecs_all = [], []
for i in range(model_c.n_components):
    kw, sc = get_topic_embedding(model_c, i)
    embeddings = embed_keywords(tuple(kw))  # tuple for lru_cache compatibility
    c_keywords_all.append(kw)
    c_vecs_all.append(np.average(embeddings, axis=0, weights=sc).reshape(1, -1))

m_keywords_all, m_vecs_all = [], []
for j in range(model_m.n_components):
    kw, sc = get_topic_embedding(model_m, j)
    embeddings = embed_keywords(tuple(kw))  # tuple for lru_cache compatibility
    m_keywords_all.append(kw)
    m_vecs_all.append(np.average(embeddings, axis=0, weights=sc).reshape(1, -1))

# -----------------------------
# Print top correspondences
# -----------------------------
def print_topics_with_full_weights(n_top=10, top_n_correspondences=5):
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
        print(f"  Classical chars: {' '.join(c_keywords_all[i])}")
        print(f"  Modern chars:    {' '.join(m_keywords_all[j])}\n")

    # All classical topics sorted by weight
    topic_weights_c = sorted([(i, W_classical[:, i].mean()) for i in range(model_c.n_components)],
                             key=lambda x: x[1], reverse=True)
    print("=== Classical Topics (sorted by weight) ===\n")
    for rank, (i, w) in enumerate(topic_weights_c):
        print(f"Classical Topic {rank+1} (weight {w:.4f})")
        print(f"  Top chars: {' '.join(c_keywords_all[i])}\n")

    # All modern topics sorted by weight
    topic_weights_m = sorted([(i, W_modern[:, i].mean()) for i in range(model_m.n_components)],
                             key=lambda x: x[1], reverse=True)
    print("=== Modern Topics (sorted by weight) ===\n")
    for rank, (i, w) in enumerate(topic_weights_m):
        print(f"Modern Topic {rank+1} (weight {w:.4f})")
        print(f"  Top chars: {' '.join(m_keywords_all[i])}\n")


print_topics_with_full_weights()
plot_topics_3d(c_vecs_all, m_vecs_all, c_keywords_all, m_keywords_all)
