# -*- coding: utf-8 -*-
"""
Created on Mon May 31 09:39:48 2021

@author: rapha
"""

from nltk.corpus import gutenberg
from transformers import AutoModelWithLMHead, AutoTokenizer
import pandas as pd
import numpy as np
import re
import nltk
import warnings
warnings.filterwarnings('ignore')

words = gutenberg.words("melville-moby_dick.txt")
raw = gutenberg.raw("melville-moby_dick.txt")
sents = gutenberg.sents("melville-moby_dick.txt")
sents = sents[4:]


from operator import itemgetter 
wpt = nltk.WordPunctTokenizer()
stop_words = nltk.corpus.stopwords.words('english')

def normalize_sents(doc):
    # adapted from https://github.com/dipanjanS/text-analytics-with-python/blob/master/New-Second-Edition/Ch04%20-%20Feature%20Engineering%20for%20Text%20Representation/Ch04a%20-%20Feature%20Engineering%20Text%20Data%20-%20Traditional%20Strategies.ipynb
    # lower case and remove special characters\whitespaces    
    liste = []
    for sen in doc:
        s = " ".join(sen)
        doc = re.sub(r'[^a-zA-Z\s]', '', s, re.I|re.A)
        doc = doc.lower()
        doc = doc.strip()
        # tokenize document
        tokens = wpt.tokenize(doc)
        # filter stopwords out of document
        #filtered_tokens = [token for token in tokens if token not in stop_words]
        filtered_tokens = tokens
        # re-create document from filtered tokens
        doc = ' '.join(filtered_tokens)
        liste.append(doc)
    return liste

norm_sentences = normalize_sents(sents)
sel_norm_sen = norm_sentences

tokenizer = AutoTokenizer.from_pretrained("t5-small")
model = AutoModelWithLMHead.from_pretrained("t5-small")

text = sel_norm_sen[50:75]
max_length = 150

input_ids = tokenizer.encode(". ".join(text), return_tensors="pt", add_special_tokens=True)

generated_ids = model.generate(input_ids=input_ids, num_beams=2, max_length=max_length, 
                               repetition_penalty=2.5, length_penalty=1.0, early_stopping=True)

preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]

# Tokenization of the first 25 sentences of moby dick: 
"""'he was ever dusting his old grammars with a queer handkerchief mockingly 
embellished with all the gay flags. he loved to dust his old grammars it somehow 
mildly reminded him of his mortality. while you take in hand to school others and
 to teach them by what name a whale fish is to be called in our tongue.'"""


##### text rank:

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

tv = TfidfVectorizer(min_df=0., max_df=1., use_idf=True)
dt_matrix = tv.fit_transform(text)
dt_matrix = dt_matrix.toarray()

vocab = tv.get_feature_names()
td_matrix = dt_matrix.T
print(td_matrix.shape)

df_vocab = pd.DataFrame(np.round(td_matrix, 2), index=vocab).head(10)

similarity_matrix = np.matmul(dt_matrix, dt_matrix.T)

import networkx

similarity_graph = networkx.from_numpy_array(similarity_matrix)

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
networkx.draw_networkx(similarity_graph, node_color='lime')

scores = networkx.pagerank(similarity_graph)
ranked_sentences = sorted(((score, index) for index, score in scores.items()), reverse=True)
ranked_sentences[:10]
num_sentences = 8
top_sentence_indices = [ranked_sentences[index][1] for index in range(num_sentences)]
top_sentence_indices.sort()

print('\n'.join(np.array(text)[top_sentence_indices]))

# Text rank is extractive, does not generate new sentences
"""
and what thing soever besides cometh within the chaos of this monster s mouth be it beast boat or stone down it goes all incontinently that foul great swallow of his and perisheth in the bottomless gulf of his paunch
holland s plutarch s morals
the indian sea breedeth the most and the biggest fishes that are among which the whales and whirlpooles called balaene take up as much in length as four acres or arpens of land
scarcely had we proceeded two days on the sea when about sunrise a great many whales and other monsters of the sea appeared
this came towards us open mouthed raising the waves on all sides and beating the sea before him into a foam
he visited this country also with a view of catching horse whales which had bones of very great value for their teeth of which he brought some to the king
and whereas all the other things whether beast or vessel that enter into the dreadful gulf of this monster s whale s mouth are immediately lost and swallowed up the sea gudgeon retires into it in great security and there sleeps
"""