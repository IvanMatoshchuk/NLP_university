from dataclasses import dataclass
from typing import List


@dataclass
class Tag:
    word: str
    tag: str


Tags = List[Tag]


@dataclass
class RawTags:
    original_sentences: List
    nltk: List
    treetagger: List
    stanza: List
    pattern: List
    spacy: List


@dataclass
class TagsToCompare:
    nltk: Tags
    treetagger: Tags
    stanza: Tags
    pattern: Tags
    spacy: Tags
