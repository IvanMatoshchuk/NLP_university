"""
Created Date : 05-Jun-21
@author : Aman Jain
"""
import nltk
from nltk.tokenize import *
##Tokenization

def file_read(lang) :
    if lang == 'en':
        filename = 'University/Exercise1/data/Harry_en.txt'
    if lang == 'de':
        filename = 'University/Exercise1/data/Harry_de.txt'
    with open(filename, encoding='utf-8') as reader:
        file: str = reader.read()
        return file

## English
contents = file_read('en')
print(word_tokenize(contents))
print(sent_tokenize(contents))

## German
contents = file_read('de')
print(word_tokenize(contents))
print(sent_tokenize(contents))




