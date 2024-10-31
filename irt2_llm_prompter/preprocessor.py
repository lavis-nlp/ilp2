import re


def remove_stopwords(s: str, stopwords) -> str:
    for stopword in stopwords:
        s = re.sub(rf"\b{stopword}\b", "", s).strip()
    s = re.sub(r"\s+", " ", s)
    return s
