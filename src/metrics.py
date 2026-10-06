import re

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Lexical proxies, not trained classifiers
TOXICITY_TERMS = ["hate", "violence", "discriminat", "racism", "sexism", "offensive"]
STEREOTYPE_PATTERNS = ["all .* are", "typical", "always", "never", "every .* is"]

_vader = SentimentIntensityAnalyzer()


def sentiment_score(text):
    return (_vader.polarity_scores(text)["compound"] + 1) / 2


def toxicity_score(text):
    text = text.lower()
    k = sum(1 for term in TOXICITY_TERMS if term in text)
    return min(k / 10.0, 1.0)


def stereotype_score(text):
    text = text.lower()
    k = sum(1 for pattern in STEREOTYPE_PATTERNS if re.search(pattern, text))
    return min(k / 5.0, 1.0)


def add_scores(df):
    df = df.copy()
    df["sentiment_score"] = df["response"].map(sentiment_score)
    df["toxicity_score"] = df["response"].map(toxicity_score)
    df["stereotype_score"] = df["response"].map(stereotype_score)
    return df
