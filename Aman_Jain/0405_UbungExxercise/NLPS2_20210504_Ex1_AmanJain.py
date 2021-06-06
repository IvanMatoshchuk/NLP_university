"""
Created Date : 14-May-21
@author : Aman Jain
"""

## Importing the Modules
import nltk
import numpy as np
from nltk.corpus import gutenberg

fileids = gutenberg.fileids()
print(fileids)

##
bible = nltk.corpus.gutenberg.words('bible-kjv.txt')
print(bible)
bigram = nltk.bigrams(bible)


##
def train_bigram_model(cfd, word, num):
    for i in np.arange(num):
        print(word, end=' ')
        word = cfd[word].max()


cfd = nltk.ConditionalFreqDist(bigram)


##
train_bigram_model(cfd, 'light', 15)


##
train_bigram_model(cfd, 'lord', 10)
