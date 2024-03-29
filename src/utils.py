import re


def preprocess_text(text):

    cleaned_text = re.sub(r"[^a-zA-Z0-9\s]", "", text)

    try:
        cleaned_text = cleaned_text.encode("iso-8859-1").decode("utf-8")
    except Exception:
        pass

    cleaned_text = " ".join(cleaned_text.split())

    cleaned_text = cleaned_text.lower()

    return cleaned_text
