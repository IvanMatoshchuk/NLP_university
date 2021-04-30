# %%
import os
import pathlib
import string
import sys

import gensim
import nltk
import numpy as np
import pandas as pd
import spacy
import treetaggerwrapper

from pattern.en import parse, split
from pattern.en import pprint


from utils.preprocess import clean_text, read_text, read_text_splitted

# %%

# %reload_ext autoreload
# %autoreload 2


# %%


lang = "en"
path_to_text = r"D:\Uni\Master\FU Data Science\Lectures\SS_21\NLP\project\task_1\data\Harry_en.txt"
read_saved = True

text = read_text_splitted(path_to_text)
labeled_pos = pd.read_csv("tokens_POS_labeled.csv", index_col=0)


# %%

# prepare text

last_5_sent_splitted = text[-5:]

last_5_sent_full, last_5_sent_full_clean = clean_text(last_5_sent_splitted)
print("Clean text:\n\n", last_5_sent_full_clean)


# %%


# tagging NLTK

print("NLTK tokenization...\n")

tags_nltk = nltk.word_tokenize(last_5_sent_full_clean)
print(f"Number of tokens: {len(tags_nltk)}")

if not read_saved:
    pd.DataFrame({"Tokens": tags_nltk}).to_csv("tokens_initial.csv")


if len(tags_nltk) != labeled_pos.shape[0]:
    print("inconsistency between nltk pos and labeled pos")


# %%

# Part of Speech NLTK


pos_nltk = []

for pos in nltk.pos_tag(tags_nltk, lang="eng"):
    pos_nltk.append(pos[1])

labeled_pos["NLTK_pos_pred"] = pos_nltk


# %%

# Part of Speech TreeTagger


tagger = treetaggerwrapper.TreeTagger(
    TAGDIR=r"D:\Uni\Master\FU Data Science\Lectures\SS_21\NLP\TreeTagger", TAGLANG=lang,
)

print(f"TreeTagger tokenization...\n")
tags = tagger.tag_text(last_5_sent_full_clean)
tags_treetagger = [w.split("\t")[1] for w in tags]
print(f"Number of tokens: {len(tags_treetagger)}")

if len(tags_treetagger) != labeled_pos.shape[0]:
    print("inconsistency between nltk pos and labeled pos")

labeled_pos["TreeTag_pos_pred"] = tags_treetagger


# %%

# Part of Speech Spacy

nlp = spacy.load("en_core_web_sm")

doc = nlp(last_5_sent_full_clean)

tokens_pos = []
tokens_dep = []

for token in doc:
    tokens_pos.append(token.pos_)
    tokens_dep.append(token.dep_)

if len(tokens_pos) != labeled_pos.shape[0]:
    print("inconsistency between spacy pos and labeled pos")

labeled_pos["Spacy_pos_pred"] = tokens_pos

# %%

# Pattern

s = parse(last_5_sent_full_clean)
s = split(s)

pattern_pos = []

for i in range(len(s)):
    pattern_pos = pattern_pos + list(s.sentences[i].pos)

if len(pattern_pos) != labeled_pos.shape[0]:
    print("inconsistency between pattern pos and labeled pos")

labeled_pos["Pattern_pos_pos_pred"] = tokens_pos

# %%

[] + ["a"]


# %%
print(labeled_pos)

labeled_pos.to_csv("pos_full_pred.csv", index=False)

