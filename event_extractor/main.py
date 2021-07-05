from event import EventExtractor
e = EventExtractor()
raw_text = e.dataLoader('./data.txt')
sents = e.get_sents(raw_text)
print(e.greedy(sents))
