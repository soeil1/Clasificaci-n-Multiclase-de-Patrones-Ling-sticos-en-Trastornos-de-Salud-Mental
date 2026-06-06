"""Limpieza y tokenizacion del texto."""

import re

STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "as", "at", "be", "because", "been", "before", "being", "below",
    "between", "both", "but", "by", "can", "did", "do", "does", "doing", "down",
    "during", "each", "few", "for", "from", "further", "had", "has", "have",
    "having", "he", "her", "here", "hers", "herself", "him", "himself", "his",
    "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me",
    "more", "most", "my", "myself", "no", "nor", "now", "of", "off", "on",
    "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over",
    "own", "s", "same", "she", "should", "so", "some", "such", "t", "than",
    "that", "the", "their", "theirs", "them", "themselves", "then", "there",
    "these", "they", "this", "those", "through", "to", "too", "under", "until",
    "up", "very", "was", "we", "were", "what", "when", "where", "which", "while",
    "who", "whom", "why", "will", "with", "you", "your", "yours", "yourself",
    "yourselves", "im", "ive", "dont", "doesnt", "didnt", "ill", "id", "youre",
}
# Nota: dejamos "never", "nothing", "alone", "empty", "tired" fuera de stopwords
# porque son palabras clave para detectar depresion/suicidio.

_URL_RE      = re.compile(r"http\S+|www\.\S+")
_MENTION_RE  = re.compile(r"[@#]\w+")
_NONALPHA_RE = re.compile(r"[^a-z\s]")
_MULTISPACE  = re.compile(r"\s+")


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = _URL_RE.sub(" ", text)
    text = _MENTION_RE.sub(" ", text)
    text = _NONALPHA_RE.sub(" ", text)
    text = _MULTISPACE.sub(" ", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    tokens = text.split()
    return [tok for tok in tokens if tok not in STOPWORDS and len(tok) > 2]


def preprocess(text: str) -> list[str]:
    return tokenize(clean_text(text))


if __name__ == "__main__":
    demo = "I've been SO tired and empty for 3 weeks... http://x.co @user nothing helps!!!"
    print("Crudo :", demo)
    print("Limpio:", clean_text(demo))
    print("Tokens:", preprocess(demo))
