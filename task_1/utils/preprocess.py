def read_text(path_to_text: str) -> str:
    
    with open(path_to_text, "r", encoding="utf-8") as f:
        text = f.read()

    return text.replace("\n","")


def read_text_splitted(path_to_text: str) -> str:
    
    text = []
    
    with open(path_to_text, "r", encoding="utf-8") as f:
        for line in f:
            line_clean = line.replace("\n",'')
            if len(line_clean) < 2:
                continue
            text.append(line_clean)

    return text