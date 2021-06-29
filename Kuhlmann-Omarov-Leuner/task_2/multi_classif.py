# -*- coding: utf-8 -*-
"""
Created on Mon May 17 16:43:21 2021

@author: rapha
# classify 20 newsgroups datasets
"""


import sklearn
import nltk

import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics

from sklearn.datasets import fetch_20newsgroups
from pprint import pprint

def evaluate_pred(targets, pred):
    acc = metrics.balanced_accuracy_score(targets, pred)
    pre = metrics.precision_score(targets, pred, average='weighted')
    rec = metrics.recall_score(targets, pred, average='weighted')
    f1 = metrics.f1_score(targets, pred, average='weighted')
    return [acc, pre, rec, f1]

newsgroups_train = fetch_20newsgroups(subset='train')
newsgroups_test = fetch_20newsgroups(subset='train')

unique_IDs, counts = np.unique(newsgroups_train.target, return_counts = True)
plt.bar(unique_IDs, counts)

vectorizer = TfidfVectorizer()
vectors_train = vectorizer.fit_transform(newsgroups_train.data)
vectors_test = vectorizer.transform(newsgroups_test.data)

targets_train = newsgroups_train.target
targets_test = newsgroups_test.target

# use metrics: 
# balanced accuracy score (unbalanced dataset)
# precision
# recall
# f1 score: 2*(precision*recall)/(precision+recall)

# NAIVE Bayes
from sklearn.naive_bayes import MultinomialNB

bayes = MultinomialNB(alpha = .01)
bayes.fit(vectors_train, targets_train)
bayes_pred = bayes.predict(vectors_test)
bayes_eval = evaluate_pred(targets_test, bayes_pred)

# SVM 
from sklearn.svm import SVC
svclass = SVC(gamma = "auto")
svclass.fit(vectors_train, targets_train)
sv_class_pred = svclass.predict(vectors_test)
sv_class_eval = evaluate_pred(targets_test, sv_class_pred)

# Random Forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(vectors_train, targets_train)
rf_pred = rf.predict(vectors_test)
rf_eval = evaluate_pred(targets_test, rf_pred)

# ADABOOST
from sklearn.ensemble import AdaBoostClassifier
adb = AdaBoostClassifier()
adb.fit(vectors_train, targets_train)
adb_pred = adb.predict(vectors_test)
adb_eval = evaluate_pred(targets_test, adb_pred)

# knn classification
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(vectors_train, targets_train)
knn_pred = knn.predict(vectors_test)
knn_eval = evaluate_pred(targets_test, knn_pred)


results = np.array([
    bayes_eval, sv_class_eval, rf_eval, adb_eval, knn_eval])

labels = ["bayes", "svm", "randomforest", "adaboost", "knn"]

fig, axs = plt.subplots(2,2, figsize = (10,8))
axs[0,0].bar(labels, results[:,0])
axs[0,0].set_title("Balanced accuracy")
axs[0,1].bar(labels, results[:,1])
axs[0,1].set_title("Precision")
axs[1,0].bar(labels, results[:,2])
axs[1,0].set_title("Recall")
axs[1,1].bar(labels, results[:,3])
axs[1,1].set_title("F1 score")

# using word2vec
import gensim.downloader as api
wv = api.load('word2vec-google-news-300')

def tokenize(senlist):
    all_sentences = []
    all_tokens = []

    for line in newsgroups_train.data: 
        sen = line.replace('"','').replace("\n", "")
        all_sentences.append(sen)
            
        #nltk_tags
        tokens = nltk.word_tokenize(sen, language = "english")
        all_tokens.append(tokens)
    return all_sentences, all_tokens
from sklearn.preprocessing import MinMaxScaler

mm_scaler = MinMaxScaler()

def vectorize(model, sentences):

    not_vectorizable = []
    all_vects = []
    for sen in sentences:
        all_sen = []
        for word in sen:
            try:
                wordvec = wv[word]
                all_sen.append(wordvec)
            except KeyError:
                not_vectorizable.append(word)
        all_sen = np.array(all_sen)
        all_vects.append(all_sen)
    return all_vects, not_vectorizable

def senmean(sentences):
    return [np.mean(mm_scaler.fit_transform(elem), axis = 0) for elem in sentences]
                
#tokenize sentences    
sen_train, tok_train = tokenize(newsgroups_train.data)
sen_test, tok_test = tokenize(newsgroups_train.data)

#vectorize sentences
vect_train, train_nv = vectorize(wv, tok_train)
vect_test, test_nv = vectorize(wv, tok_test)

#mean sentences
vect_train_m = senmean(vect_train)
vect_test_m = senmean(vect_test)



#rerun classification
vectors_train = vect_train_m
vectors_test = vect_test_m

bayes = MultinomialNB(alpha = .01)
bayes.fit(vectors_train, targets_train)
bayes_pred = bayes.predict(vectors_test)
bayes_eval = evaluate_pred(targets_test, bayes_pred)

svclass = SVC(gamma = "auto")
svclass.fit(vectors_train, targets_train)
sv_class_pred = svclass.predict(vectors_test)
sv_class_eval = evaluate_pred(targets_test, sv_class_pred)

rf = RandomForestClassifier()
rf.fit(vectors_train, targets_train)
rf_pred = rf.predict(vectors_test)
rf_eval = evaluate_pred(targets_test, rf_pred)

adb = AdaBoostClassifier()
adb.fit(vectors_train, targets_train)
adb_pred = adb.predict(vectors_test)
adb_eval = evaluate_pred(targets_test, adb_pred)

knn = KNeighborsClassifier()
knn.fit(vectors_train, targets_train)
knn_pred = knn.predict(vectors_test)
knn_eval = evaluate_pred(targets_test, knn_pred)

results = np.array([
    bayes_eval, sv_class_eval, rf_eval, adb_eval, knn_eval])

labels = ["bayes", "svm", "randomforest", "adaboost", "knn"]

fig, axs = plt.subplots(2,2, figsize = (10,8))
axs[0,0].bar(labels, results[:,0])
axs[0,0].set_title("Balanced accuracy")
axs[0,1].bar(labels, results[:,1])
axs[0,1].set_title("Precision")
axs[1,0].bar(labels, results[:,2])
axs[1,0].set_title("Recall")
axs[1,1].bar(labels, results[:,3])
axs[1,1].set_title("F1 score")
