import string


def read_text(path_to_text: str) -> str:

    with open(path_to_text, "r", encoding="utf-8") as f:
        text = f.read()

    return text.replace("\n", "")


def read_text_splitted(path_to_text: str) -> str:

    text = []

    with open(path_to_text, "r", encoding="utf-8") as f:
        for line in f:
            line_clean = line.replace("\n", "")
            if len(line_clean) < 2:
                continue
            text.append(line_clean)

    return text


def clean_text(text: str) -> list:

    text_full = " ".join(text)
    text_full_clean = "".join(
        [i for i in text_full if i not in string.punctuation.replace(".", "").replace("!", "") + "”"]
    )
    text_full_clean = (
        text_full_clean.replace("That’ll", "That will")
        .replace("Potter’s", "Potter is")
        .replace("Voldy’s", "Voldy has")
        .replace("let’s", "let us")
    )

    return text_full, text_full_clean
