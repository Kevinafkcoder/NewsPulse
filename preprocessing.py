
import re
import html
import pandas as pd

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


NEWS_STOPWORDS = {
    "said", "says", "according", "reported", "report",
    "reports", "news", "today", "yesterday", "latest",
    "currently", "recently", "also", "one", "two",
    "year", "years", "day", "days", "week", "weeks",
    "month", "months", "time", "times", "newspaper",
    "article", "articles", "read", "watch", "video",
    "videos", "click", "image", "images"
}

STOPWORDS = set(ENGLISH_STOP_WORDS).union(NEWS_STOPWORDS)


def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = html.unescape(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    words = text.split()

    cleaned_words = [
        word for word in words
        if not word.isdigit()
        and word not in STOPWORDS
        and len(word) > 1
    ]

    return " ".join(cleaned_words)


def normalize_title(title):
    """Normalize titles for exact duplicate detection."""

    if not isinstance(title, str):
        return ""

    title = html.unescape(title).lower()
    title = re.sub(r"[^a-z0-9\s]", " ", title)
    title = re.sub(r"\s+", " ", title).strip()

    return title


def prepare_articles(df):
    df = df.copy()

    for column in ["title", "description", "content"]:
        if column not in df.columns:
            df[column] = ""

        df[column] = df[column].fillna("").astype(str)

    # Remove exact duplicate titles
    df["normalized_title"] = df["title"].apply(
        normalize_title
    )

    before = len(df)

    df = df.drop_duplicates(
        subset=["normalized_title"],
        keep="first"
    ).copy()

    exact_removed = before - len(df)

    # Give titles extra importance
    df["raw_text"] = (
        df["title"] + " " +
        df["title"] + " " +
        df["description"] + " " +
        df["content"]
    )

    df["clean_text"] = df["raw_text"].apply(clean_text)

    df = df[
        df["clean_text"].str.strip() != ""
    ].copy()

    df = df.reset_index(drop=True)

    print("Articles before deduplication:", before)
    print("Exact duplicates removed:", exact_removed)
    print("Articles after exact deduplication:", len(df))

    return df
