

def read_text(path_to_text: str) -> str:
    
    with open(path_to_text, "r", encoding="utf-8") as f:
        text = f.read()

    return text



