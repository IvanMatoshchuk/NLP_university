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
from utils.postprocess import save_xlsx

# %%

# %reload_ext autoreload
# %autoreload 2


project_path = pathlib.Path(__file__).parent

print(project_path)

# %%

# arguments

lang = "en"
path_to_text = os.path.join(project_path, "data", "Harry_en.txt")
read_saved = True
treeTagger_path = os.path.join(project_path.parent.parent, "TreeTagger")


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

nltk_pos = pd.concat([labeled_pos, pd.DataFrame(pos_nltk, columns=["NLTK_pos_pred"])], axis=1)

# labeled_pos["NLTK_pos_pred"] = pos_nltk



# %%

# Part of Speech TreeTagger


tagger = treetaggerwrapper.TreeTagger(TAGDIR=treeTagger_path, TAGLANG=lang,)

print(f"TreeTagger tokenization...\n")
tags = tagger.tag_text(last_5_sent_full_clean)
tags_treetagger = [w.split("\t")[1] for w in tags]
print(f"Number of tokens: {len(tags_treetagger)}")

if len(tags_treetagger) != labeled_pos.shape[0]:
    print("inconsistency between nltk pos and labeled pos")


TreeTagger_pos = pd.concat([labeled_pos, pd.DataFrame(tags_treetagger, columns=["TreeTag_pos_pred"])], axis=1)

# labeled_pos["TreeTag_pos_pred"] = tags_treetagger


# %%

# Part of Speech Spacy

nlp = spacy.load("en_core_web_sm")

doc = nlp(last_5_sent_full_clean)

tokens_pos = []
tokens_dep = []
tokens_pos_full = []

for token in doc:
    tokens_pos.append(token.pos_)
    tokens_dep.append(token.dep_)

    tokens_pos_full.append(spacy.explain(token.pos_))

if len(tokens_pos) != labeled_pos.shape[0]:
    print("inconsistency between spacy pos and labeled pos")

# labeled_pos["Spacy_pos_pred"] = tokens_pos
# labeled_pos["Spacy_pos_pred_full"] = tokens_pos_full

spacy_pos = pd.concat(
    [labeled_pos, pd.DataFrame({"Spacy_pos_pred": tokens_pos, "Spacy_pos_full_pred": tokens_pos_full})], axis=1
)


# %%

# Pattern

s = parse(last_5_sent_full_clean)
s = split(s)

pattern_pos = []

for i in range(len(s)):
    pattern_pos.extend(list(s.sentences[i].pos))

if len(pattern_pos) != labeled_pos.shape[0]:
    print("inconsistency between pattern pos and labeled pos")

#labeled_pos["Pattern_pos_pred"] = pattern_pos

pattern_pos = pd.concat(
    [labeled_pos, pd.DataFrame({"Pattern_pos_pred": pattern_pos})], axis=1
)



# %%
print(labeled_pos.shape)

save_xlsx([nltk_pos,TreeTagger_pos,spacy_pos,pattern_pos],
["nltk_pos","TreeTagger_pos","spacy_pos","pattern_pos"])

#labeled_pos.to_csv("pos_full_pred.csv", index=False)
print("Generated pos_full_pred.csv\n")

# print("\nPreview:")
# labeled_pos.head()



# %%
# Chunks pattern


s = parse(last_5_sent_full_clean)
pprint(s)


# %%
