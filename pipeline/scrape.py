"""
scrape.py
---------
Fetches app metadata and reviews from Google Play Store.
"""

import string
import pandas as pd
from google_play_scraper import app, search, reviews

from pipeline.config import REVIEW_COUNT, REVIEW_LANG, REVIEW_COUNTRY, NUM_COMPETITORS


def strip_title(title: str) -> str:
    """Return the first word of an app title, stripped of punctuation.
    e.g. 'Duolingo: Language Lessons' -> 'Duolingo'
    """
    return title.split()[0].translate(str.maketrans("", "", string.punctuation))


def get_app_metadata(app_id: str) -> dict:
    """Fetch metadata for a single app from Google Play."""
    result = app(app_id)
    return result


def parse_metadata(result: dict) -> dict:
    """Extract the fields we actually need from a raw google-play-scraper result."""
    return {
        "Title": strip_title(result.get("title", "Unknown")),
        "FullTitle": result.get("title", "Unknown"),
        "Genre": result.get("genre", "Unknown"),
        "GenreId": result.get("genreId", "Unknown"),
        "Description": result.get("description", ""),
        "DateReleased": result.get("released", "Unknown"),
        "LastUpdatedDate": result.get("lastUpdatedOn", "Unknown"),
        "NumInstalls": int(result.get("realInstalls", 0)),
        "NumReviews": int(result.get("reviews", 0)),
        "Score": float(result.get("score", 0)),
        "NumRatings": int(result.get("ratings", 0)),
        "AdSupported": result.get("adSupported", False),
        "Price": result.get("price", 0),
        "ScoreHistogram": result.get("histogram", []),
    }


def get_competitor_ids(result: dict, n: int = NUM_COMPETITORS) -> list[str]:
    """Search for competitors based on the app title and return their app IDs."""
    title = strip_title(result["title"])
    search_results = search(title)
    return [
        x["appId"]
        for x in search_results[1 : n + 1]
        if title.lower() not in x["appId"]
    ]


def get_reviews(app_id: str) -> pd.DataFrame:
    """Fetch reviews for an app and return as a DataFrame."""
    review_results, _ = reviews(
        app_id,
        count=REVIEW_COUNT,
        lang=REVIEW_LANG,
        country=REVIEW_COUNTRY,
    )
    review_df = pd.DataFrame(review_results)
    review_df["n_words"] = review_df["content"].str.split().str.len()
    return review_df


def get_competitor_metadata(app_id: str, competitor_ids: list[str]) -> pd.DataFrame:
    """Fetch and parse metadata for main app and all competitors."""
    rows = []
    for cid in [app_id] + competitor_ids:
        result = get_app_metadata(cid)
        row = parse_metadata(result)
        row["AppId"] = cid
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("NumInstalls", ascending=False)
    return df
