# %%
import os
import sys
import pathlib

import nltk
import string


# %%

# %reload_ext autoreload
# %autoreload 2


from utils.preprocess import read_text, read_text_splitted

# %%


lang = "en"

path_to_text = r"D:\Uni\Master\FU Data Science\Lectures\SS_21\NLP\project\task_1\data\Harry_en.txt"

text = read_text_splitted(path_to_text)


# %%


last_5_sent_splitted = text[-5:]
last_5_sent_full = " ".join(last_5_sent_splitted)
last_5_sent_full_clean = "".join([i for i in last_5_sent_full if i not in string.punctuation.replace(".", "").replace("!","") + "”"])
last_5_sent_full_clean = (
    last_5_sent_full_clean.replace("That’ll", "That will")
    .replace("Potter’s", "Potter is")
    .replace("Voldy’s", "Voldy has")
    .replace("let’s", "let us")
)


print(last_5_sent_full_clean)

manual_tokens = []

for sent in last_5_sent_splitted:

    # sent = sent.translate(str.maketrans('', '',string.punctuation))

    sent = "".join([i for i in sent if i not in string.punctuation.replace(".", "")])  # leave dots

    # sent = sent.replace("’ll", " will")  # .replace("’s")

    sen_splitted = sent.split(" ")
    manual_tokens.append(sen_splitted)

# %%

sum([len(i) for i in manual_tokens])
manual_tokens
# %%


# tagging length

print("NLTK tagging...\n")


tags_nltk = nltk.word_tokenize(last_5_sent_full_clean)
tags_nltk_clean = nltk.word_tokenize(last_5_sent_full_clean)
print(f"Number of tags with punct: {len(tags_nltk)}")
print(f"Number of tags without punct: {len(tags_nltk_clean)}")
tags_nltk_clean


# %%
pos_nltk = []

for pos in nltk.pos_tag(tags_nltk_clean, lang="eng"):
    pos_nltk.append(pos)

pos_nltk

# %%
nltk.word_tokenize("That'll be")
# %%
