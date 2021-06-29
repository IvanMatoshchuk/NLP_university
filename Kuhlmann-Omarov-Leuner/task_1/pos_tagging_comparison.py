import json

import nltk
import treetaggerwrapper

import stanza
import spacy

from entities import RawTags, TagsToCompare, Tag
from project_utils import turn_on_parse
from utils import read_file, delete_substrings, strs, ints, filter_indices, print_table, flatten

def read_harry(sentences: ints, lang = "en") -> strs:
    if lang == "en":
        filename = "data/Harry_en.txt"
    if lang == "de":
        filename = "data/Harry_de.txt"
    result = [delete_substrings(l, ['",', '"','»','«']) for l in read_file(filename) if l]
    return filter_indices(result, sentences)


sentences_to_analyze = [3, 19, 86, 100, 124]
# sentences_to_analyze = [124]
harry_en = read_harry(sentences_to_analyze, lang = "en")
harry_de = read_harry(sentences_to_analyze, lang = "de")

treetagger_en = treetaggerwrapper.TreeTagger(TAGLANG='en')
treetagger_de = treetaggerwrapper.TreeTagger(TAGLANG='de')

stanza_en = stanza.Pipeline('en')
stanza_de = stanza.Pipeline('de')


spacy_nlp_en = spacy.load('en_core_web_sm')
spacy_nlp_de = spacy.load('de_core_news_sm')


parse_en = turn_on_parse(lang = "en")
parse_de = turn_on_parse(lang = "de")



def fill_english_taggers() -> RawTags:
    result = RawTags([], [], [], [], [], [])

    for line in harry_en:
        result.original_sentences.append(line)
        result.nltk.append(nltk.pos_tag(nltk.word_tokenize(line, language="english")))
        result.treetagger.append(treetaggerwrapper.make_tags(treetagger_en.tag_text(line)))
        result.stanza.append(stanza_en(line))
        result.pattern.append(parse_en(line).split())
        for token in spacy_nlp_en(line):
            result.spacy.append([token.text, token.pos_, token.dep])

    return result

def fill_german_taggers() -> RawTags:
    result = RawTags([], [], [], [], [], [])

    for line in harry_de:
        result.original_sentences.append(line)
        result.nltk.append(nltk.pos_tag(nltk.word_tokenize(line, language="german")))
        result.treetagger.append(treetaggerwrapper.make_tags(treetagger_de.tag_text(line)))
        result.stanza.append(stanza_de(line))
        result.pattern.append(parse_de(line).split())
        for token in spacy_nlp_de(line):
            result.spacy.append([token.text, token.pos_, token.dep])

    return result

def unify_tags_format(raw_tags: RawTags) -> TagsToCompare:
    result_nltk = [Tag(tag[0], tag[1]) for tag in flatten(raw_tags.nltk)]
    result_treetagger = [Tag(tag.word, tag.pos) for tag in flatten(raw_tags.treetagger)]
    result_stanza = [Tag(tag['text'], tag['upos']) for tag in flatten(flatten(json.loads(str(raw_tags.stanza))))]
    result_pattern = [Tag(tag[0], tag[1]) for tag in flatten(flatten(raw_tags.pattern))]
    result_spacy = [Tag(tag[0], tag[1]) for tag in raw_tags.spacy]

    return TagsToCompare(result_nltk, result_treetagger, result_stanza, result_pattern, result_spacy)


def compare_postags(tags_to_compare: TagsToCompare):
    table = [['Word', 'nltk', 'treetagger', 'stanza', 'pattern', 'spacy'], []]
    for i in range(len(tags_to_compare.nltk)):
        word = tags_to_compare.nltk[i].word
        table.append([word,
                      tags_to_compare.nltk[i].tag,
                      tags_to_compare.treetagger[i].tag,
                      tags_to_compare.stanza[i].tag,
                      tags_to_compare.pattern[i].tag,
                      tags_to_compare.spacy[i].tag,
                      ])

        if '.' in word:
            table.append([])

    print_table(table)


raw_tags_en = fill_english_taggers()
tags_to_compare_en = unify_tags_format(raw_tags_en)
compare_postags(tags_to_compare_en)

raw_tags_de = fill_german_taggers()
tags_to_compare_de = unify_tags_format(raw_tags_de)
compare_postags(tags_to_compare_de)

