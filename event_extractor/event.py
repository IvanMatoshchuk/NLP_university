from __future__ import unicode_literals, print_function
from Giveme5W1H.extractor.document import Document
from Giveme5W1H.extractor.extractor import MasterExtractor
import spacy
from spacy.lang.en import English
import hashlib
import json
import numpy as np


class EventExtractor():

    def __init__(self):
        self.ex = MasterExtractor()

    def getWho(self, doc):
        try:
            return doc.get_top_answer('who').get_parts_as_text()
        except:
            return 'not_found'

    def getWhere(self, doc):
        try:
            return doc.get_top_answer('where').get_parts_as_text()
        except:
            return 'not_found'

    def getWhen(self, doc):
        try:
            return doc.get_top_answer('when').get_parts_as_text()
        except:
            return 'not_found'

    def getWhat(self, doc):
        try:
            return doc.get_top_answer('what').get_parts_as_text()
        except:
            return 'not_found'

    def dataLoader(self, filepath):
        with open(filepath, mode='r', encoding='utf8') as f:
            data = f.read().replace('\n', '')
        return data

    def get_sents(self, text):
        nlp = English()
        nlp.add_pipe('sentencizer')
        doc = nlp(text)
        return [sent.text for sent in doc.sents]

    def getEvent(self, doc):
        doc = self.ex.parse(doc)
        functions = [self.getWho, self.getWhat, self.getWhere, self.getWhen]
        answers = [fn(doc) for fn in functions]
        count = sum([x != 'not_found' for x in answers])
        if count > 0:
            self.print_events(answers)
        return self.hash('_'.join(answers)), count

    def print_events(self, answers):
        pp = ['{}:{}'.format(q, a) for q, a in zip(
            ['who', 'what', 'where', 'when'], answers)]
        with open('./results.txt', 'a', encoding="utf8") as f:
            json.dump(pp, f)
            f.write('\n')

        print(pp)

    def hash(self, text):
        return int(hashlib.sha1(text.encode("utf-8")).hexdigest(), 16) % (10 ** 8)

    # simple sent by sent event counter
    def events(self, sents):
        seen = {}
        for sent in sents:
            doc = Document.from_text(text=sent)
            event, eventLen = self.getEvent(doc)
            # no answers were found
            if eventLen < 1:
                continue
        if event not in seen:
            seen[event] = True
        return len(seen.keys())

    def greedy(self, sents):
        sents_so_far = ''
        seen = {}
        for sent in sents:
            sents_so_far += sent
            doc = Document.from_text(text=sents_so_far)
            event, eventLen = self.getEvent(doc)

            # minimal context needed for 1 event
            if eventLen > 0:
                if event not in seen:
                    seen[event] = True
                # clear current context and start again
                sents_so_far = ''
        with open('./results.txt', 'a') as f:
            f.write("number of events: {}".format(len(seen.keys())))

        return len(seen.keys())
