"""
preprocess.py
-------------
Review filtering, language detection, and stopword construction.
"""

import re
import numpy as np
import pandas as pd
from langdetect import detect
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from pipeline.config import (
    MIN_WORD_COUNT,
    MAX_SCORE,
    GENERIC_APP_WORDS,
    GENRE_ACTIVITY_WORDS,
)


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------

def is_english(text: str) -> bool:
    """Return True if text is detected as English."""
    try:
        return detect(text) == "en"
    except Exception:
        return False


def filter_language(review_df: pd.DataFrame) -> pd.DataFrame:
    """Keep only English reviews."""
    mask = review_df["content"].apply(is_english)
    return review_df[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Review filtering
# ---------------------------------------------------------------------------

def filter_reviews(
    review_df: pd.DataFrame,
    min_word_count: int = MIN_WORD_COUNT,
    max_score: int = MAX_SCORE,
) -> pd.DataFrame:
    """Filter to negative reviews meeting minimum word count."""
    return review_df[
        (review_df["n_words"] >= min_word_count)
        & (review_df["score"] <= max_score)
    ].reset_index(drop=True)


def to_text_list(review_df: pd.DataFrame) -> list[str]:
    """Convert review content column to a clean list of strings."""
    return [str(t) for t in review_df["content"]]


# ---------------------------------------------------------------------------
# Stopword construction
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    """Lowercase and extract alpha tokens."""
    return re.findall(r"[a-z]+", text.lower())


def title_lemma_and_plural(title: str) -> list[str]:
    """
    Extract tokens from the descriptive part of an app title
    (after the colon/dash), plus naive singular/plural variants.
    e.g. 'Duolingo: Language Lessons' -> ['duolingo', 'language', 'lesson',
                                           'languages', 'lessons']
    """
    parts = re.split(r"[:\-]", title, maxsplit=1)
    app_name = parts[0].strip().lower()
    words = tokenize(parts[1]) if len(parts) > 1 else []

    variants = []
    for word in words:
        if word.endswith("s"):
            variants.append(word[:-1])   # lessons -> lesson
        else:
            variants.append(word + "s")  # language -> languages

    return [app_name] + words + variants


def build_stopwords(result: dict) -> list[str]:
    """
    Build an app-specific stopword list from:
      1. App title tokens (name + descriptive words + plural/singular variants)
      2. Genre-derived activity verbs
      3. Generic app vocabulary
      4. sklearn English stopwords
    """
    title_words = title_lemma_and_plural(result.get("title", ""))
    genre_words = GENRE_ACTIVITY_WORDS.get(result.get("genreId", ""), [])

    return list(set(
        title_words
        + genre_words
        + GENERIC_APP_WORDS
        + list(ENGLISH_STOP_WORDS)
    ))
