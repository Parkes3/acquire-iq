"""
themes.py
---------
Assigns BERTopic topics to predefined themes via Claude API,
and builds topic-level and theme-level summary tables.
"""

import json
import numpy as np
import pandas as pd
from bertopic import BERTopic

from config import THEMES, CLAUDE_MODEL, PARETO_CUTOFF


# ---------------------------------------------------------------------------
# Theme assignment
# ---------------------------------------------------------------------------

def assign_themes(
    topic_labels: list[str],
    themes: list[str] = THEMES,
    client = None,
    model: str = CLAUDE_MODEL,
) -> list[dict]:
    """
    Send topic labels to Claude and return theme assignments.
    Returns a list of dicts: [{'topic_index': 0, 'theme': '...'}, ...]
    """
    labels_str = "\n".join(f"{i}: {label}" for i, label in enumerate(topic_labels))

    prompt = f"""Assign each topic to exactly one theme.

Themes: {', '.join(themes)}

Topics:
{labels_str}

Return ONLY valid JSON:
{{"assignments": [{{"topic_index": 0, "theme": "..."}}]}}"""

    response = client.messages.create(
        model=model,
        max_tokens=500,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )

    if response.stop_reason == "max_tokens":
        raise ValueError("Theme assignment response truncated — increase max_tokens")

    text = (
        response.content[0].text
        .strip()
        .replace("```json", "")
        .replace("```", "")
    )
    return json.loads(text)["assignments"]


# ---------------------------------------------------------------------------
# Table construction
# ---------------------------------------------------------------------------

def extract_claude_label(cell) -> str:
    """Safely extract the Claude label string from a BERTopic representation cell."""
    if isinstance(cell, list) and len(cell) > 0:
        item = cell[0]
        if isinstance(item, tuple):
            return item[0]
        return str(item)
    return str(cell)


def build_topic_table(
    topic_model: BERTopic,
    review_df_filtered: pd.DataFrame,
    client,
    themes: list[str] = THEMES,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build topic-level and theme-level summary tables.

    Returns:
        topic_table  — one row per topic, sorted by Severity
        theme_table  — one row per theme, aggregated counts
        review_df    — review_df_filtered with Topic, TopicName, Theme columns added
    """
    info = topic_model.get_topic_info()
    final_topics = topic_model.topics_

    # Extract Claude labels
    info["claude_label"] = info["Claude"].apply(extract_claude_label)

    # Build lookup maps
    topic_num_to_name = info.set_index("Topic")["Name"]
    topic_name_to_claude = info.set_index("Name")["claude_label"]

    # Assign themes (exclude outlier topic -1)
    info_no_outliers = info[info["Topic"] != -1].reset_index(drop=True)
    assignments = assign_themes(
        info_no_outliers["claude_label"].tolist(), themes, client
    )
    assignment_df = pd.DataFrame(assignments)
    info_no_outliers["Theme"] = assignment_df["theme"].values
    topic_to_theme = dict(zip(info_no_outliers["Name"], info_no_outliers["Theme"]))

    # Map topics onto reviews
    review_df = review_df_filtered.copy()
    review_df["Topic"] = final_topics
    review_df["TopicName"] = review_df["Topic"].map(topic_num_to_name)
    review_df["ClaudeLabel"] = review_df["TopicName"].map(topic_name_to_claude)
    review_df["Theme"] = review_df["TopicName"].map(topic_to_theme)

    # Topic-level aggregation
    topic_table = (
        review_df.groupby("TopicName")
        .agg(
            numReviews=("reviewId", "count"),
            numThumbsUp=("thumbsUpCount", "sum"),
            avgRating=("score", "mean"),
        )
    )

    # Severity score: weighted combination of volume, engagement, and low rating
    topic_table["thumbsAndReviews"] = (
        topic_table["numThumbsUp"] / 2 + topic_table["numReviews"]
    )
    topic_table["review_normalized"] = (
        topic_table["numReviews"] / topic_table["numReviews"].max()
    )
    topic_table["thumbsup_normalized"] = (
        topic_table["numThumbsUp"] / topic_table["numThumbsUp"].max()
    )
    topic_table["invRating"] = (5 - topic_table["avgRating"]) / 4
    topic_table["Severity"] = (
        topic_table["review_normalized"] * 0.4
        + topic_table["thumbsup_normalized"] * 0.3
        + topic_table["invRating"] * 0.3
    )

    topic_table["Claude"] = topic_table.index.map(topic_name_to_claude)
    topic_table["Theme"] = topic_table.index.map(topic_to_theme)
    topic_table = topic_table.sort_values("Severity", ascending=False)

    # Pareto cumulative columns
    topic_table["CumulativeThumbsAndReviews"] = topic_table["thumbsAndReviews"].cumsum()
    topic_table["CumulativePercentage"] = (
        topic_table["CumulativeThumbsAndReviews"]
        / topic_table["thumbsAndReviews"].sum()
    )

    # Theme-level aggregation
    theme_table = (
        topic_table.groupby("Theme")
        .agg(
            numTopics=("Theme", "count"),
            numReviews=("numReviews", "sum"),
            numThumbsUp=("numThumbsUp", "sum"),
            avgRating=("avgRating", "mean"),
        )
    )
    theme_table["percentReviews"] = (
        theme_table["numReviews"] / theme_table["numReviews"].sum()
    )
    theme_table["percentThumbsUp"] = (
        theme_table["numThumbsUp"] / theme_table["numThumbsUp"].sum()
    )
    theme_table = theme_table.sort_values("numReviews", ascending=False)

    return topic_table, theme_table, review_df
