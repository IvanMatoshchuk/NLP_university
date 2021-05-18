# 20 Newsgroup dataset classification

# %%

import re
import os
import sys

from typing import List

import nltk
from nltk.featstruct import Feature
from nltk.sem.logic import ExpectedMoreTokensException
import gensim
import gensim.downloader as api

import pandas as pd
import numpy as np

from nltk import tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.datasets import fetch_20newsgroups
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

stop_words = stopwords.words("english")
lemmatizer = WordNetLemmatizer()

wv = api.load("glove-wiki-gigaword-100")

# %%

# %%


# %%

#####
# Create dataframe
#####

print("\nCreating dictionary!\n")
output_dict = {}
j = 0
for i in os.listdir("120"):

    output_dict[i] = {}

    for book in os.listdir(os.path.join("120", i)):

        if book.startswith("."):
            continue

        df_book = pd.read_csv(os.path.join("120", i, book))
        df_book["text"] = df_book["text"].apply(lambda x: eval(x))

        output_dict[i][book] = set(sum(df_book["text"], []))

        j += 1
        sys.stdout.write(f" {j}/120   {book} PROCESSED!\n")


# %%

print("\nCreating dataframe\n")


arr_index = np.array([i for i in output_dict.keys()])

arr_book = [j for i in output_dict.keys() for j in output_dict[i].keys()]


arr_text = np.array([output_dict[i].values() for i in arr_book])

len(arr_book)

# %%
df_final = pd.DataFrame(index=np.repeat(arr_index, repeats=20), columns=["book", "text"])

df_final["book"] = arr_book
df_final["genre"] = df_final.index
df_final.reset_index(drop=True, inplace=True)


def return_value(a, b):

    return list(output_dict[a][b])


df_final["text"] = df_final.apply(lambda x: return_value(x.genre, x.book), axis=1)

df_final.to_csv("preprocessed.csv", index=False)

# %%

#####################################
#  READ SAVED DATAFRAME
#####################################

df_final = pd.read_csv("preprocessed.csv")
df_final["text"] = df_final["text"].apply(lambda x: eval(x.lower()))

# for i in df_final["text"].apply(lambda x: len(x)):
#     print(i)

# %%

#########################
# embedding and finding min, max, mean and var of a long vector of embedding (concatenate all embeddings)
#########################


output = {"mean": [], "max": [], "min": [], "var": [], "std": []}

for idx, corpus in enumerate(df_final["text"]):
    j = 0
    w2v_list = []

    for word in corpus:
        try:
            w2v_list.append(wv[word])
        except Exception as e:
            # print(e)
            j += 1

    print(np.round(j / len(corpus), 2) * 100, "% of words not embedded")

    output["mean"].append(np.mean(w2v_list, axis=0))
    output["max"].append(np.max(w2v_list, axis=0))
    output["min"].append(np.min(w2v_list, axis=0))
    output["var"].append(np.var(w2v_list, axis=0, ddof=1))
    output["std"].append(np.std(w2v_list, axis=0))


# %%

#######################
#  CREATE DF OF FEATURES
#######################

df_output = pd.DataFrame.from_dict(output)


df_output_features = []

for col in ["mean", "var", "std"]:

    _df = pd.DataFrame([pd.Series(x) for x in df_output[col]])
    _df.columns = [f"{col}_{x+1}" for x in _df.columns]

    df_output_features.append(_df)

df_output_features = pd.concat(df_output_features, axis=1)

df_output_features.shape

df_output_features.head()

# %%

df_procesed = pd.concat([df_final, df_output_features], axis=1)


df_procesed.shape
# %%
df_procesed.head()


# %%

#########################
##### Training
#########################
df_test = df_procesed.sample(20, random_state=1234)
print("Test shape: ", df_test.shape)

df_train = df_procesed[~df_procesed.index.isin(list(df_test.index))]
print("Train shape: ", df_train.shape)


feature_cols = [i for i in df_train.columns if i.startswith(("mean", "var"))]
feature_cols

X_train, y_train = df_train[feature_cols], df_train["genre"]
X_test, y_test = df_test[feature_cols], df_test["genre"]


models = (
    RandomForestClassifier(n_estimators=1000, max_depth=10),
    AdaBoostClassifier(n_estimators=100),
    KNeighborsClassifier(),
)

for model in models:

    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    print("\n\n************************************************************")
    print(f"\n{model.__class__.__name__} classification report:\n")

    print("for train data:")
    print(classification_report(y_train, y_pred_train, zero_division=1))

    print("\nfor test data:")
    print(classification_report(y_test, y_pred_test, zero_division=1))


# %%

######################


# %%


# %%


# ##################################################
# ##################################################
# ##################################################
# from sklearn.decomposition import TruncatedSVD


# df_test = df_final.sample(10, random_state=1)
# print("Test shape: ", df_test.shape)

# df_train = df_final[~df_procesed.index.isin(list(df_test.index))]
# print("Train shape: ", df_train.shape)


# from tqdm import tqdm


# embeddings_all = np.zeros((df_final.shape[0], 10000))

# svd = TruncatedSVD(100)

# for row in tqdm(range(df_final.shape[0])):

#     embeddings = np.zeros((len(df_final["text"][0]), 100))

#     for i, w in enumerate(df_final["text"][0]):
#         try:
#             embeddings[i] = wv[w]
#         except Exception as e:

#             continue

#     # remove zero entries
#     idx = np.argwhere(np.all(embeddings[..., :] == 0, axis=1))
#     embeddings = np.delete(embeddings, idx, axis=0)

#     embeddings_lsa = svd.fit_transform(embeddings.T)

#     explained_variance = svd.explained_variance_ratio_.sum()
#     print("  Explained variance of the SVD step: {}%".format(int(explained_variance * 100)))

#     embeddings_all[row, :] = embeddings_lsa.reshape(1, -1)


# # %%


# # %%
# # split train - test

# df_train = df_final.sample(105, random_state=321)

# embeddings_train = embeddings_all[df_train.index, :]

# print("Train shape: ", df_train.shape)
# df_test = df_final[~df_final.index.isin(list(df_train.index))]
# embeddings_test = embeddings_all[df_test.index, :]
# print("Test shape: ", df_test.shape)


# # %%
# model = KNeighborsClassifier(3)

# model.fit(embeddings_train, df_train["genre"])

# # %%

# # %%
# predictions = model.predict(embeddings_test)

# # %%
# (predictions == df_test["genre"]).mean()  # ¯\_(ツ)_/¯

# # better then random guess !

# # %%
