
manual_tokens = []

for sent in last_5_sent_splitted:

    # sent = sent.translate(str.maketrans('', '',string.punctuation))

    sent = "".join([i for i in sent if i not in string.punctuation.replace(".", "")])  # leave dots

    # sent = sent.replace("’ll", " will")  # .replace("’s")

    sen_splitted = sent.split(" ")
    manual_tokens.append(sen_splitted)

# %%

print(f"Tokens manually: {sum([len(i) for i in manual_tokens])}")
manual_tokens