import os
import pandas as pd
import json
import streamlit as st

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "outputs")
SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "sample")

def safe_id(app_id: str) -> str:
    return app_id.replace(".", "_")


@st.cache_data
def load_app_data(app_id: str) -> dict:
    """Load pre-computed parquet files for an app. Cached so reruns are instant."""
    sid = safe_id(app_id)
    return {
        "topic_table": pd.read_parquet(f"{OUTPUT_DIR}/{sid}_topics.parquet"),
        "theme_table":  pd.read_parquet(f"{OUTPUT_DIR}/{sid}_themes.parquet"),
        "review_df":    pd.read_parquet(f"{OUTPUT_DIR}/{sid}_reviews.parquet"),
    }

@st.cache_data
def load_groups(output_dir: str) -> list[dict]:
    groups_path = f"{output_dir}/groups.json"
    if not os.path.exists(groups_path):
        return []
    with open(groups_path) as f:
        return json.load(f)

@st.cache_data
def get_apps_in_group(main_app_id: str, output_dir: str) -> pd.DataFrame:
    """Load metadata for all apps in a named group."""
    groups = load_groups(output_dir)
    group = next((g for g in groups if g["main_app_id"] == main_app_id), None)
    if group is None:
        return pd.DataFrame()

    rows = []
    for app_id in group["app_ids"]:
        sid = app_id.replace(".", "_")
        meta_path = f"{output_dir}/{sid}_metadata.parquet"
        if os.path.exists(meta_path):
            meta = pd.read_parquet(meta_path).iloc[0]
            rows.append({
                "app_id": app_id,
                "Title": meta.get("Title", app_id),
                "NumInstalls": int(meta.get("NumInstalls", 0)),
                "Score": float(meta.get("Score", 0)),
                "AdSupported": meta.get("AdSupported", False),
                'DateReleased': meta.get('DateReleased', ''),
                'NumReviews':meta.get('NumReviews', 0),
                "is_main": app_id == group["main_app_id"],
            })
        else:
            rows.append({
                "app_id": app_id,
                "Title": app_id,
                "NumInstalls": 0,
                "Score": 0,
                "AdSupported": None,
                "is_main": app_id == group["main_app_id"],
            })

    return pd.DataFrame(rows).sort_values("NumInstalls", ascending=False).reset_index(drop=True)


#TODO separate output dir and sample dir

def find_app_data(app_id: str) -> str | None:
    """
    Return the directory containing data for an app.
    Checks outputs first (user-generated), falls back to sample.
    Returns None if not found in either location.

    TO REPLACE OUTPUT_DIR
    """
    sid = app_id.replace(".", "_")
    for directory in [OUTPUT_DIR, SAMPLE_DIR]:
        if os.path.exists(f"{directory}/{sid}_themes.parquet"):
            return directory
    return None

