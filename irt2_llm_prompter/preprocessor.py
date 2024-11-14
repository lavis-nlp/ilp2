import re
import Stemmer


def remove_stopwords(s: str, stopwords) -> str:
    for stopword in stopwords:
        s = re.sub(rf"\b{stopword}\b", "", s).strip()
    s = re.sub(r"\s+", " ", s)
    return s

def stemming(s: str) -> str:
    stemmer = Stemmer.Stemmer('english')
    wordlist = s.split(" ")
    stemmed_wordlist = stemmer.stemWords(wordlist)
    stemmed_mentions = " ".join(stemmed_wordlist)
    return stemmed_mentions
