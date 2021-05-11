# -*- coding: utf-8 -*-
"""
Created on Tue May  4 14:50:42 2021

@author: rapha

Word generation on the bible
"""
import nltk
import numpy as np
from nltk.corpus import gutenberg
from nltk.text import Text
import random
#nltk.download('gutenberg')

thebible = gutenberg.words('bible-kjv.txt')
text = nltk.corpus.gutenberg.raw('bible-kjv.txt')
tob = Text(thebible)

#thebible =  [elem.remove(["\n", ""])]
#seed = 42
seed = random.random()

gentext1 = tob.generate(length = 100, text_seed = "god", random_seed = random.random())
print("\n")
gentext2 = tob.generate(length = 100, text_seed = "house", random_seed = random.random())
print("\n")
gentext3 = tob.generate(length = 100, text_seed = "holy", random_seed = random.random())
print("\n")


############################# Task 3

from nltk.corpus import brown
brown.categories()
cats = ['adventure', 'belles_lettres', 'editorial', 'fiction', 'government', 'hobbies',
'humor', 'learned', 'lore', 'mystery', 'news', 'religion', 'reviews', 'romance',
'science_fiction']

religion = brown.tagged_words(categories='religion')

romance = brown.tagged_words(categories='romance')

from nltk.corpus import wordnet as wn
type = 'n'

#synsets = wn.all_synsets(type)

def polysemy(word): 
    return wn.synsets(word)

def polysemy_noun(word): 
    return wn.synsets(word, "n")


average_pol_rel = np.mean(np.array([len(polysemy(elem[0])) for elem in religion]))
average_pol_rom = np.mean(np.array([len(polysemy(elem[0])) for elem in romance]))
print("Average polysemy religion: {}, average polysemy romance: {}".format(average_pol_rel, average_pol_rom))

average_pol_nouns_rel = np.mean(np.array([len(polysemy(elem[0])) for elem in religion if elem[1]=="NN"]))
average_pol_nouns_rel2 = np.mean(np.array([len(polysemy_noun(elem[0])) for elem in religion if elem[1]=="NN"]))

average_pol_nouns_rom = np.mean(np.array([len(polysemy(elem[0])) for elem in religion if elem[1]=="NN"]))
average_pol_nouns_rom2 = np.mean(np.array([len(polysemy_noun(elem[0])) for elem in religion if elem[1]=="NN"]))


print("Average polysemy religion nouns: {}, Average polysemy only taking into account noun meanings of the noun: {}".format(average_pol_nouns_rel, average_pol_nouns_rel2))




############################ Task 2
n_meronyms = 0
n_synsets = 0
for word in set(religion):
    if word[1]=="NN":
        for synset in wn.synsets(word[0], "n"):
            n_synsets += 1
            n_meronyms += len(synset.part_meronyms())
            n_meronyms += len(synset.substance_meronyms())

print(n_meronyms)
            
n_meronyms = 0
n_synsets = 0
for word in set(romance):
   # print(word)
    if word[1]=="NN":
        #print(word[0])
        for synset in wn.synsets(word[0], "n"):
           # print(synset)
            n_synsets += 1
          #  print(synset.part_meronyms())
            n_meronyms += len(synset.part_meronyms())
            n_meronyms += len(synset.substance_meronyms())
            
print(n_meronyms)

# Religion vs Romantic
# For sets: 1021 vs 2089
# For whole DS: 6683 vs 14228
            





def explain(term):
    """Get a description for a given POS tag, dependency label or entity type.
    term (str): The term to explain.
    RETURNS (str): The explanation, or `None` if not found in the glossary.
    EXAMPLE:
        >>> spacy.explain(u'NORP')
        >>> doc = nlp(u'Hello world')
        >>> print([w.text, w.tag_, spacy.explain(w.tag_) for w in doc])
    """
    if term in GLOSSARY:
        return GLOSSARY[term]
    
print(explain('DET'))
GLOSSARY = {
    # POS tags
    # Universal POS Tags
    # http://universaldependencies.org/u/pos/
    "ADJ": "adjective",
    "ADP": "adposition",
    "ADV": "adverb",
    "AUX": "auxiliary",
    "CONJ": "conjunction",
    "CCONJ": "coordinating conjunction",
    "DET": "determiner",
    "INTJ": "interjection",
    "NOUN": "noun",
    "NUM": "numeral",
    "PART": "particle",
    "PRON": "pronoun",
    "PROPN": "proper noun",
    "PUNCT": "punctuation",
    "SCONJ": "subordinating conjunction",
    "SYM": "symbol",
    "VERB": "verb",
    }
 