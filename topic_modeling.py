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

@lru_cache(maxsize=None)
def embed_word(word):
    return embedding_model.encode(word)

def embed_keywords(words):
    return np.array([embed_word(w) for w in words])

rcParams['font.sans-serif'] = ['Arial Unicode MS']
rcParams['axes.unicode_minus'] = False

# remove unknown characters that cannot be read
def clean_text(tokens):
    text = "".join(tokens)
    text = re.sub(r"[^\u4e00-\u9fff]", "", text)
    return text

classical_docs = [clean_text(tokens) for tokens in classical_poems]
modern_docs    = [clean_text(tokens) for tokens in modern_poems]

embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

model_c = KeyNMF(n_components=15, encoder=embedding_model, top_n=15, random_state=42,
                 vectorizer=CountVectorizer(analyzer="char", ngram_range=(1,2), min_df=1))
model_m = KeyNMF(n_components=15, encoder=embedding_model, top_n=15, random_state=42,
                 vectorizer=CountVectorizer(analyzer="char", ngram_range=(1,2), min_df=1))

# -----------------------------
# Batched fitting
# -----------------------------
def batched(iterable, n):
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(itertools.islice(it, n)):
        yield batch

def extract_and_save_keywords(model, docs, filename):
    with Path(filename).open("w") as f:
        for batch in batched(docs, 200):
            batch_keywords = model.extract_keywords(list(batch))
            for kw in batch_keywords:
                f.write(json.dumps(kw) + "\n")

def stream_keywords(filename):
    with Path(filename).open() as f:
        for line in f:
            yield json.loads(line.strip())

def fit_from_keywords(model, filename, epochs=3):
    for epoch in range(epochs):
        for batch in batched(stream_keywords(filename), 200):
            model.partial_fit(keywords=list(batch))

print("Extracting classical keywords...")
extract_and_save_keywords(model_c, classical_docs, "keywords_c.jsonl")
print("Extracting modern keywords...")
extract_and_save_keywords(model_m, modern_docs, "keywords_m.jsonl")

print("Fitting classical model...")
fit_from_keywords(model_c, "keywords_c.jsonl")
print("Fitting modern model...")
fit_from_keywords(model_m, "keywords_m.jsonl")

# -----------------------------
# Extract top keywords per topic
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
        items = list(zip(words[:n], scores[:n]))

    else:
        raise ValueError(f"Unexpected topic format: {type(topic)}")

    words = [w for w, _ in items]
    if len(words) == 0:
        raise ValueError("Empty topic encountered")

    scores = np.array([s for _, s in items])
    if scores.sum() == 0:
        scores = np.ones_like(scores) / len(scores)
        print("SUM OF SCORES IS 0!!!!")
    else:
        scores = scores / scores.sum()

    return words, scores

def get_top_keywords(model, topic_idx, n=10):
    topics = model.get_topics()
    words, _ = parse_topic(topics[topic_idx], n)
    return words
# -----------------------------
# Topic weights (document-topic matrix)
# -----------------------------
W_classical = model_c.transform(classical_docs)
W_modern    = model_m.transform(modern_docs)

# normalize
W_classical_norm = W_classical / (W_classical.sum(axis=1, keepdims=True) + 1e-9)
W_modern_norm    = W_modern / (W_modern.sum(axis=1, keepdims=True) + 1e-9)

# -----------------------------
# Compare topics using keyword embedding similarity
# -----------------------------
def get_topic_embedding(model, topic_idx, n=10):
    topics = model.get_topics()
    keywords, scores = parse_topic(topics[topic_idx], n)
    return keywords, scores

# Precompute once
c_keywords_all, c_vecs_all = [], []
for i in range(model_c.n_components):
    kw, sc = get_topic_embedding(model_c, i)
    embeddings = embed_keywords(kw)
    c_keywords_all.append(kw)
    c_vecs_all.append(np.average(embeddings, axis=0, weights=sc).reshape(1, -1))

m_keywords_all, m_vecs_all = [], []
for j in range(model_m.n_components):
    kw, sc = get_topic_embedding(model_m, j)
    embeddings = embed_keywords(kw)
    m_keywords_all.append(kw)
    m_vecs_all.append(np.average(embeddings, axis=0, weights=sc).reshape(1, -1))

# -----------------------------
# Print top correspondences + all topics
# -----------------------------
def print_topics_with_full_weights(n_top=10, top_n_correspondences=5):
    similarities = []
    for i in range(model_c.n_components):
        for j in range(model_m.n_components):
            score = cosine_similarity(c_vecs_all[i], m_vecs_all[j])[0][0]
            c_weight = W_classical_norm[:, i].mean()
            m_weight = W_modern_norm[:, j].mean()
            similarities.append((score, i, j, c_weight, m_weight))

    similarities.sort(key=lambda x: x[0], reverse=True)
    top_correspondences = similarities[:top_n_correspondences]

    print(f"=== Top {top_n_correspondences} Classical→Modern Topic Correspondences (embedding cosine similarity) ===\n")
    for rank, (score, i, j, c_weight, m_weight) in enumerate(top_correspondences):
        c_kw = "".join(c_keywords_all[i])
        m_kw = "".join(m_keywords_all[j])
        print(f"#{rank+1} Cosine Similarity: {score:.4f}")
        print(f"  Classical Topic {i+1} (weight {c_weight:.4f}) → Modern Topic {j+1} (weight {m_weight:.4f})")
        print(f"  Classical chars: {c_kw}")
        print(f"  Modern chars:    {m_kw}\n")

    topic_weights_c = [(i, W_classical_norm[:, i].mean()) for i in range(model_c.n_components)]
    topic_weights_c.sort(key=lambda x: x[1], reverse=True)
    print("=== Classical Topics (sorted by weight) ===\n")
    for rank, (i, mean_weight) in enumerate(topic_weights_c):
        print(f"Classical Topic {rank+1} (weight {mean_weight:.4f})")
        print(f"  Top characters: {''.join(c_keywords_all[i])}\n")

    topic_weights_m = [(i, W_modern_norm[:, i].mean()) for i in range(model_m.n_components)]
    topic_weights_m.sort(key=lambda x: x[1], reverse=True)
    print("=== Modern Topics (sorted by weight) ===\n")
    for rank, (i, mean_weight) in enumerate(topic_weights_m):
        print(f"Modern Topic {rank+1} (weight {mean_weight:.4f})")
        print(f"  Top characters: {''.join(m_keywords_all[i])}\n")

print_topics_with_full_weights(n_top=10, top_n_correspondences=5)