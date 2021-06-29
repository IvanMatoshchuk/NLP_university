# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 21:19:22 2021

@author: rapha
"""
# -*- coding: iso-8859-1 -*-

import nltk
import codecs
import os
import treetaggerwrapper
import numpy as np
import spacy
from project_utils import turn_on_parse
from utils import read_file, delete_substrings, strs, ints, filter_indices, print_table, flatten

import stanza

def read_harry(sentences: ints, lang = "en") -> strs:
    if lang == "en":
        filename = "data/Harry_en.txt"
    if lang == "de":
        filename = "data/Harry_de.txt"
    result = [delete_substrings(l, ['",', '"','»','«']) for l in read_file(filename) if l]
    if len(sentences) == 0:
        res = result
    else: 
        res = filter_indices(result, sentences)
    return res

sentences_to_analyze = []

file_en = read_harry(sentences_to_analyze, lang = "en")
file_de = read_harry(sentences_to_analyze, lang = "de")


#tagger_de = treetaggerwrapper.TreeTagger(TAGLANG='en')

sentences_de = []
sentences_en = []


# treetagger
treetagger_de = treetaggerwrapper.TreeTagger(TAGLANG='de')
treetagger_en = treetaggerwrapper.TreeTagger(TAGLANG='en')

parse_de = turn_on_parse(lang = "de")
parse_en = turn_on_parse(lang = "en")


#stanza.download("de")
#stanza.download('en')  
stanza_en = stanza.Pipeline('en') 
stanza_de = stanza.Pipeline('de')


spacy_nlp_en = spacy.load('en_core_web_sm')
spacy_nlp_de = spacy.load('de_core_news_sm')



# Large Spacy model downloads:
#        python -m spacy download de_dep_news_trf
#        python -m spacy download en_core_web_trf

# spacy_nlp_en = spacy.load("en_core_web_trf")
# spacy_nlp_de = spacy.load("de_dep_news_trf")


tokens_de = []

tags_de_nltk = []
tags_en_nltk = []

chunks_de_nltk = []

tags_de_tt = []
tags_en_tt = []

tags_de_tt = []
tags_en_tt = []

tags_en_stanza = []
tags_de_stanza = []

tags_en_pattern = []
tags_de_pattern = []

tags_en_spacy = []
tags_de_spacy = []



tokens_en = []

# German 
for line in file_de: 
    if line != '\n':
        sen = line.replace('"','').replace("\n", "")
        sentences_de.append(sen)
        
        #nltk_tags
        tokens = nltk.word_tokenize(sen, language = "german")
        tokens_de.append(tokens)
        tags_de_nltk.append(nltk.pos_tag(tokens))
        
        
        #tt_tags
        tags = treetagger_de.tag_text(sen)
        final_tags = treetaggerwrapper.make_tags(tags)
        tags_de_tt.append(final_tags)

        #stanza tags
        tags_de_stanza.append(stanza_de(sen))
        
        
        #pattern tags
        pat = parse_de(sen, relations = True, lemmata = True).split()
        tags_de_pattern.append(pat)

        
        #spacy tags
        toksen = spacy_nlp_de(sen)
        tags_de_spacy.append([[token.text, token.pos_, token.dep ] for token in toksen])



# English
for line in file_en: 
    if line != '\n':
        sen = line.replace('"','').replace("\n", "")
        sentences_en.append(sen)
        
        #nltk_tags
        tokens = nltk.word_tokenize(sen, language = "english")
        tokens_en.append(tokens)
        tags_en_nltk.append(nltk.pos_tag(tokens))
        
        #tt_tags
        tags = treetagger_en.tag_text(sen)
        final_tags = treetaggerwrapper.make_tags(tags)
        tags_en_tt.append(final_tags)
        
        #stanza tags
        tags_en_stanza.append(stanza_en(sen))

        
        #pattern tags
        pat = parse_en(sen).split()
        tags_en_pattern.append(pat)
        
        #spacy tags
        toksen = spacy_nlp_en(sen)
        tags_en_spacy.append([token.text, token.pos_, token.dep ] for token in toksen)

        
# ANALYSIS PART: see other python document

# PHRASES            
# get nounphrases using spacy:    
def extractNP(text):
    doc = spacy_nlp_de(text)
    result = []
    for nc in doc.noun_chunks:
        result.append(nc.text)
    return result

NP = []
for sen in sentences_de:
    NP.append(extractNP(sen))
    
# count NP:
print("The total number of NP in the dataset according to spacy is: {}".format(len(sum(NP, []))))


# chunking for nltk data
# POS Tagging results from NLTK seem to be really bad!!!
# from https://pythonprogramming.net/community/42/NLTK%20parsing/
chunkGram = r"""
            S: {Q}
            Q:  {<NP><VP>|<NP><VP><ADJP>}
            NP: {<DT><NN>|<NN>|<PRO>|<PRO><NP>|<NN><PREP><NP>|<NN><CONJ><NN>|<NN><NP>|<PREP><NP>|<NP><VP>|<NUM><NP>|<ADJ><NP>}       # Chunk sequences of DT, JJ, NN
            PRO: {<PRP$>|<PRP$><JJ>}
            VP: {<VBZ>|<VBZ><VP>|<VP><NP>|<VB>|<VB><NP>|<VBD>|<VBD><NP>}
            ADJP: {<ADJP><NP>|<JJ><ADJP>|<JJ>}
            PREP:{<IN>|<IN><NP>}
            ADV: {<RB><VP>}
            NUM:  {<CD>}
            CONJ:{<CC>}
            """
grammar_NP = """
    NP: {<NNP>*}
        {<DT>?<JJ>?<NNS>}
        {<NN><NN>}"""

            
# Number of NP in data using NLTK dataset:
def chunk(sentence, grammar):
    parser = nltk.RegexpParser(grammar)
    result = parser.parse(sentence)
    results = []
    for subtree in result.subtrees():
        if subtree.label() == 'NP':
            t = subtree
            results.append([word for word, pos in t.leaves()])
    return results

NP_nltk = []
for sen in tags_de_nltk:
    NP_nltk.append(chunk(sen, grammar_NP))
    
# count NP:
print("The total number of NP in the dataset according to nltk is: {}".format(len(sum(NP_nltk, []))))


# create phrases from pattern data

def pattern_phrases(pattern_data):
    phrases_lists = [[], [], [], [], []]
    phrases_count = [0, 0, 0, 0, 0]
    phr = ["NP", "VP", "ADJP", "ADVP", "PP"]
    phr_dict = {
        "NP": 0, "VP": 1, "ADJP": 2, "ADVP": 3, "PP": 4}
    data = pattern_data[0]
    
    relations = list(set([elem[4] for elem in data]))
    relations.remove('O')

    
    phrases_rel = [elem for elem in data if elem[4] != 'O']
    for rel in relations:
        phr_id = rel.split("-")[0]
        phrases = [elem for elem in phrases_rel if elem[4] == rel]
        if phr_id in phr:
            phrases_lists[phr_dict[phr_id]].append(phrases)
            phrases_count[phr_dict[phr_id]] += 1
        
    phrases_nonrel = [elem for elem in data if elem[4] == 'O']
    for word in phrases_nonrel:
        phrase_tok = word[2]
        phr_id = phrase_tok.split("-")[-1]
        if phr_id in phr:
            phrases_lists[phr_dict[phr_id]].append(word)
            phrases_count[phr_dict[phr_id]] += 1
    return phrases_lists, phrases_count


all_NP, all_VP, all_ADJP, all_ADVP, all_PP = [], [], [], [], []
phrase_count = np.zeros((len(tags_de_pattern), 5))
for i, sen in enumerate(tags_de_pattern):
    [NP, VP, ADJP, ADVP, PP], counter = pattern_phrases(sen)
    all_NP.append(NP)
    all_VP.append(VP)
    all_ADJP.append(ADJP)
    all_ADVP.append(ADVP)
    all_PP.append(PP)
    phrase_count[i,:] = counter
    
summed_phrases = np.sum(phrase_count, axis = 0)

print("""The total number of phrases in the dataset is :\n
      NP: {} \n VP: {} \n ADJP: {} \n ADVP: {} \n PP: {}""".format(
      summed_phrases[0],summed_phrases[1],summed_phrases[2],summed_phrases[3],summed_phrases[4], ))
        
# high differences between the estimations of noun phrases between spacy, nltk, 
            
