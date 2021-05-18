# 20 Newsgroup dataset classification

# %%

import re

from typing import List

import nltk
import pandas as pd
from nltk import tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.datasets import fetch_20newsgroups
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

stop_words = stopwords.words("english")
lemmatizer = WordNetLemmatizer()

# %%

# read data
newsgroups_train = fetch_20newsgroups(subset="train")
print("Train length: ", len(newsgroups_train["target"]))
newsgroups_test = fetch_20newsgroups(subset="test")
print("Test length: ", len(newsgroups_test["target"]))

# encode labels
label_encoder = {num: label for num, label in zip(set(newsgroups_train["target"]), newsgroups_train["target_names"])}


# %%

# preprocess text


def tokenizer(text: str) -> List[str]:

    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    tokens = word_tokenize(text)

    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return tokens


def main():

    X_train, y_train = newsgroups_train["data"], newsgroups_train["target"]
    X_test, y_test = newsgroups_test["data"], newsgroups_test["target"]

    tfidf_vect = TfidfVectorizer(tokenizer=tokenizer)

    X_train = tfidf_vect.fit_transform(X_train)
    print("Train converted to tfidf!")
    X_test = tfidf_vect.transform(X_test)
    print("Test converted to tfidf!")

    models = (RandomForestClassifier(), AdaBoostClassifier())
    for model in models:

        model.fit(X_train, y_train)

        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        print("Classification report for train data:")
        print(classification_report(y_train, y_pred_train))

        print("\nClassification report for test data:")
        print(classification_report(y_test, y_pred_test))

        pd.DataFrame(
            confusion_matrix(y_test, y_pred_test), columns=label_encoder.values(), index=label_encoder.values()
        ).to_csv(f"{model.__class__.__name__}_TFIDF.csv")

    print("\nDone!")


# # %%
# tokenized_train = list(map(lambda x: tokenizer(x), newsgroups_train["data"]))
# print("\ntrain tokenized!")


# %%

# X_train[0].toarray()[X_train[0].toarray() > 0]


# %%


# %%
if __name__ == "__main__":
    main()
