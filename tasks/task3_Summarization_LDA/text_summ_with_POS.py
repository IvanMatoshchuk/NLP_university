# %%
import os
import re
import sys
import urllib
from pathlib import Path

import nltk
import networkx
import numpy as np
from scipy.sparse.linalg import svds
import pandas as pd
from gensim.summarization import summarize
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt


chapter = 2
folder = Path(__file__).parent
path_to_text = os.path.join(folder, f"chapter_{chapter}.txt")


stop_words = nltk.corpus.stopwords.words("english")


def read_text(path_to_text: str) -> str:

    with open(path_to_text, "r", encoding="utf-8") as f:
        text = f.read()

    text = re.sub(r"\n|\r", " ", text)
    text = re.sub(r" +", " ", text)

    return text.strip()  # .replace("\n", " ")


#%%


def normalize_document(doc):
    # lower case and remove special characters\whitespaces
    # doc = re.sub(r'[^a-zA-Z\s]', '', doc, re.I|re.A)
    doc = doc.lower()
    doc = doc.strip()
    # tokenize document
    tokens = nltk.word_tokenize(doc)
    # filter stopwords out of document
    filtered_tokens = [token for token in tokens if token not in stop_words]

    filtered_tokens = [
        i
        for i, j in nltk.pos_tag(filtered_tokens, lang="eng")
        if j in ["RB", "NN", "JJ", "NN", "NNS", "VB", "VBG", "VBD", "VBZ", "NNP"]
    ]

    # re-create document from filtered tokens
    doc = " ".join(filtered_tokens)
    return doc


# %%
########################################
########################################
########################################
#         gensim summarization         #
########################################
########################################
########################################

text = read_text(path_to_text)

# print(summarize(text, ratio=0.1, split = False))

# %%

print("\n***** Summary gensim, word_count = 200 *****\n")

print(summarize(text, word_count=200, split=False))

# %%
########################################
########################################
########################################
#       Latent Semantic Analysis       #
########################################
########################################
########################################

# tokenize sentences and normalize
text = read_text(path_to_text)

print("\nStarting LSA...")
print("\nconstructing TFIDF Matrix...")

text_sentences = nltk.sent_tokenize(text)
print("\nNumber of sentences: ", len(text_sentences))
normalize_corpus = np.vectorize(normalize_document)

normalized_text = normalize_corpus(text_sentences)

print(normalized_text)

# %%
# TFIDF

tfidf = TfidfVectorizer()
dt_matrix = tfidf.fit_transform(normalized_text)

dt_matrix = dt_matrix.toarray()

vocab = tfidf.get_feature_names()
print("Length vocab: ", len(vocab))
td_matrix = dt_matrix.T
# print(td_matrix.shape)
# pd.DataFrame(np.round(td_matrix, 2), index=vocab).head(10)

# %%
# LSA


def low_rank_svd(matrix, singular_count=2):
    u, s, vt = svds(matrix, k=singular_count)
    return u, s, vt


num_sentences = 5
num_topics = 3

u, s, vt = low_rank_svd(td_matrix, singular_count=num_topics)
print("\nSVD shapes: ")
print(u.shape, s.shape, vt.shape)
term_topic_mat, singular_values, topic_document_mat = u, s, vt

# remove singular values below threshold
sv_threshold = 0.5
min_sigma_value = max(singular_values) * sv_threshold
singular_values[singular_values < min_sigma_value] = 0
# print(singular_values)

salience_scores = np.sqrt(np.dot(np.square(singular_values), np.square(topic_document_mat)))
salience_scores

# %%
# top sentences

top_sentence_indices = (-salience_scores).argsort()[:num_sentences]
top_sentence_indices.sort()

print(f"\n***** Summary LSA, num_topics = {num_topics}, num_sentences = {num_sentences} *****\n")

print("\n".join(np.array(text_sentences)[top_sentence_indices]))


# %%
########################################
########################################
########################################
#              Text Rank               #
########################################
########################################
########################################

# compute similarity matrix (mult tfidf matrix with itself)

print("\nStarting Text Rank...\n")

similarity_matrix = np.matmul(dt_matrix, dt_matrix.T)
print("Similarity Matrix: ", similarity_matrix.shape)
# np.round(similarity_matrix, 3)


similarity_graph = networkx.from_numpy_array(similarity_matrix)
# similarity_graph

plt.figure(figsize=(12, 6))
networkx.draw_networkx(similarity_graph, node_color="lime")


plt.savefig("Network_Graph.png")

# %%

scores = networkx.pagerank(similarity_graph)
ranked_sentences = sorted(((score, index) for index, score in scores.items()), reverse=True)
ranked_sentences[:10]
# %%
num_sentences = 5
top_sentence_indices = [ranked_sentences[index][1] for index in range(num_sentences)]
top_sentence_indices.sort()

print("\n***** Sumamry Tree Rank *****\n")
print("\n".join(np.array(text_sentences)[top_sentence_indices]))


# %%

from transformers import AutoModelWithLMHead, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-summarize-news")
model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-summarize-news")


def summarize(text, max_length=150):
    input_ids = tokenizer.encode(text, return_tensors="pt", add_special_tokens=True)

    generated_ids = model.generate(
        input_ids=input_ids,
        num_beams=2,
        max_length=max_length,
        repetition_penalty=2.5,
        length_penalty=1.0,
        early_stopping=True,
    )

    preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]

    return preds[0]


# %%

text = read_text()

summarize
