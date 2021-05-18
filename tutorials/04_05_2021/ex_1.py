# %%
import nltk
import random

from nltk import ngrams
from nltk.text import Text
from nltk.tag import TrigramTagger, BigramTagger

nltk.download("gutenberg")

text_raw = nltk.corpus.gutenberg.raw("bible-kjv.txt")
text_tokenized = nltk.corpus.gutenberg.words("bible-kjv.txt")

# %%

a= 3
def generate_model(cfdist, word, num=15):
    for i in range(num):
        print(word)
        word = cfdist[word].max()


bigrams = nltk.bigrams(text_tokenized)

cfd = nltk.ConditionalFreqDist(bigrams)
# %%

generate_model(cfd, "living", 15)

# %%

cfd["living"].plot(20)


# %%

tob = Text(text_tokenized)
gentext1 = tob.generate(length=100, text_seed="sea")

# %%
gentext1


# %%

nltk.download("brown")
nltk.download("universal_tagset")

brown_rel_tagged = nltk.corpus.brown.tagged_words(categories="religion", tagset="universal")
brown_rom_tagged = nltk.corpus.brown.tagged_words(categories="romance", tagset="universal")


# %%
rel_clean = sorted(
    set([word for (word, tag) in brown_rel_tagged if tag == "NOUN" and word.isalpha() and not word[0].isupper()])
)

romn_clean = sorted(
    set([word for (word, tag) in brown_rom_tagged if tag == "NOUN" and word.isalpha() and not word[0].isupper()])
)


# %%
from nltk.corpus import wordnet as wn


type = "n"

# %%
synsets = wn.all_synsets(type)


# %%
def find_polysemy(text):

    count = 0

    for w in text:
        count += len(wn.synsets(w))

    return count / len(text)


n_rel = find_polysemy(rel_clean)
print(n_rel)

n_romn = find_polysemy(romn_clean)
print(n_romn)

# %%
# meronyms

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()


def find_meronyms(text):

    count_part_m = 0
    count_subst_m = 0

    for w in text:

        try:

            word = wn.synset(f"{lemmatizer.lemmatize(w)}.n.01")

            count_part_m += len(word.part_meronyms())
            count_subst_m += len(word.substance_meronyms())

        except Exception as e:
            print(e)

    return count_part_m, count_subst_m


# %%
count_part_m, count_subst_m = find_meronyms(rel_clean)

# %%
print(count_part_m + count_subst_m)


# %%
count_part_m, count_subst_m = find_meronyms(rel_clean)

# %%
print(count_part_m + count_subst_m)


# %%


# %%
count_part_m, count_subst_m = find_meronyms(romn_clean)

print(count_part_m + count_subst_m)

# %%


# %%
q = "worries"
tree = wn.synset(f"{q}.n.01")

tree.part_meronyms()
# %%

wn.synsets("worries", wn.NOUN)


# %%
