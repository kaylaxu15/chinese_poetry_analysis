from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
import numpy as np
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from count_ancient_tokens import classical_poems
from count_modern_tokens import modern_poems
from sklearn.metrics.pairwise import cosine_similarity

# Reconstruct plain strings from your token lists
classical_docs = ["".join(tokens) for tokens in classical_poems]
modern_docs    = ["".join(tokens) for tokens in modern_poems]
all_docs = classical_docs + modern_docs

# Classical — character-level TF-IDF
vectorizer_c = TfidfVectorizer(analyzer="char", ngram_range=(1, 2), max_features=5000)
tfidf_classical = vectorizer_c.fit_transform(classical_docs)
nmf_c = NMF(n_components=8, random_state=42).fit(tfidf_classical)
feature_names_c = vectorizer_c.get_feature_names_out()

# Modern — word-level TF-IDF (jieba already tokenized)
spaced_modern = [" ".join(tokens) for tokens in modern_sentences]
vectorizer_m = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), max_features=5000)
tfidf_modern = vectorizer_m.fit_transform(spaced_modern)
nmf_m = NMF(n_components=8, random_state=42).fit(tfidf_modern)
feature_names_m = vectorizer_m.get_feature_names_out()

def top_keywords_c(i, n=10):
    return [feature_names_c[j] for j in nmf_c.components_[i].argsort()[-n:][::-1]]

def top_keywords_m(i, n=10):
    return [feature_names_m[j] for j in nmf_m.components_[i].argsort()[-n:][::-1]]

W_classical = nmf_c.transform(tfidf_classical) # weights
W_modern    = nmf_m.transform(tfidf_modern)

print("=== Classical topics ===")
for i in range(8):
    mean_weight = W_classical[:, i].mean()
    keywords = " ".join(top_keywords_c(i))
    print(f"Topic {i} (weight {mean_weight:.4f}): {keywords}")

print("\n=== Modern topics ===")
for i in range(8):
    mean_weight = W_modern[:, i].mean()
    keywords = " ".join(top_keywords_m(i))
    print(f"Topic {i} (weight {mean_weight:.4f}): {keywords}")

#===================COMPARISON OF TOPICS ================================
# Project both sets of topics into a shared character space for comparison
vectorizer_shared = TfidfVectorizer(analyzer="char", ngram_range=(1, 2), max_features=8000)
vectorizer_shared.fit(classical_docs + modern_docs)

# Get TF-IDF representation of each topic's top keywords
def topic_keyword_vector(keywords, vectorizer):
    text = " ".join(keywords)
    return vectorizer.transform([text])

print("=== Closest modern counterpart for each classical topic ===\n")
for i in range(8):
    c_keywords = top_keywords_c(i, 20)
    c_vec = topic_keyword_vector(c_keywords, vectorizer_shared)

    best_score, best_j, best_keywords = -1, -1, []
    for j in range(8):
        m_keywords = top_keywords_m(j, 20)
        m_vec = topic_keyword_vector(m_keywords, vectorizer_shared)
        score = cosine_similarity(c_vec, m_vec)[0][0]
        if score > best_score:
            best_score, best_j, best_keywords = score, j, m_keywords

    print(f"Classical {i} → Modern {best_j}  (similarity: {best_score:.4f})")
    print(f"  Classical: {' '.join(top_keywords_c(i, 8))}")
    print(f"  Modern:    {' '.join(top_keywords_m(best_j, 8))}")
    print()

