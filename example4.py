from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic
import pandas as pd
import nltk

nltk.download('wordnet')
nltk.download('wordnet_ic')

simlex = pd.read_csv("https://www.phon.ucl.ac.uk/courses/pals0039/data/SimLex-999.txt",sep='\t', lineterminator='\n')
print(simlex)

computer = wn.synset('computer.n.01')
calculator = wn.synset('calculator.n.01')


"""
path_ similarity = 1 / (dis1 + dis2 + 1)
1. path_ Similarity is the similarity of two word sets;
2. Dis1 and dis2 respectively represent the depth difference between the two word sets and their lowest common upper word set
3. The two 1s are parameters used for normalization
"""
print(wn.path_similarity(computer, calculator))


""" 
Leacock-Chodorow Similarity = -log(len(a, b) / 2 * d)
1. len(a, b) is the shortest path length and d the taxonomy depth.
2. Return a score denoting how similar two word senses are, based on the shortest path that connects the senses (as above) and the maximum depth of the taxonomy in which the senses occur.
 """
print(wn.lch_similarity(computer, calculator))

""" 
Wu-Palmer Similarity = 2 * depth(lcs(s1, s2)) / (depth(s1) + depth(s2))
1. depth(s1): The depth of synonymous set, 
2. Return a score denoting how similar two word senses are, based on the depth of the two senses in the taxonomy and that of their Least Common Subsumer (most specific ancestor node). Note that at this time the scores given do not always agree with those given by Pedersen’s Perl implementation of Wordnet Similarity.
 """
print(wn.wup_similarity(computer, calculator))



brown_ic = wordnet_ic.ic('ic-brown.dat')

""" 
res_similarity =  max IC(c), c∈LCA(a, b)
1. Resnik Similarity: Return a score denoting how similar two word senses are, based on the Information Content (IC) of the Least Common Subsumer (most specific ancestor node).
2. Note that for any similarity measure that uses information content, the result is dependent on the corpus used to generate the information content and the specifics of how the information content was created.
 """
print(computer.res_similarity(calculator, brown_ic))

""" 
jcn_similarity = 1 / (IC(s1) + IC(s2) - 2 * IC(lcs))
1. Jiang-Conrath Similarity Return a score denoting how similar two word senses are, based on the Information Content (IC) of the Least Common Subsumer (most specific ancestor node) and that of the two input Synsets.
 """
print(computer.jcn_similarity(calculator, brown_ic))


""" 
lin_similarity = 2 * IC(lcs) / (IC(s1) + IC(s2))
1. Return a score denoting how similar two word senses are, based on the Information Content (IC) of the Least Common Subsumer (most specific ancestor node) and that of the two input Synsets.
 """
print(computer.lin_similarity(calculator, brown_ic))

"""  """
""" 
In my opinion, the above word similarity calculation methods can be divided into three categories,
namely, path based method, information content-based method, and the above-mentioned mixed method of path and information content.
There is no doubt that the third method should be the best.
 """