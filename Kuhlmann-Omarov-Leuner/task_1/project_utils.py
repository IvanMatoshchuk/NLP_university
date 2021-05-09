#from pattern.text.en import parse

from entities import TagsToCompare
from utils import print_table


def turn_on_parse(lang = "en"):
    if lang == "en":
        from pattern.text.en import parse
    elif lang == "de":
        from pattern.text.de import parse
    for i in range(3):
        try:
            parse('parse goes brrrr....')
        except RuntimeError:
            print('brrr')
        else:
            return parse

    return parse


def print_words(tags_to_compare: TagsToCompare):
    table = [['nltk', 'treetagger', 'stanza', 'pattern', 'spacy'], []]
    for i in range(len(tags_to_compare.nltk)):
        try:
            table.append([tags_to_compare.nltk[i].word,
                          tags_to_compare.treetagger[i].word,
                          tags_to_compare.stanza[i].word,
                          tags_to_compare.pattern[i].word,
                          tags_to_compare.spacy[i].word,
                          ])

            if '.' in tags_to_compare.nltk[i].word:
                table.append([])

        except IndexError:
            print(f'{i} is too big')

    print_table(table)