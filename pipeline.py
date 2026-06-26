"""
pipeline.py
-----------
Top-level analyse_app function and CLI entry point.

Usage:
    python pipeline.py --app_id com.duolingo
    python pipeline.py --app_id com.babbel --output_dir data/outputs
"""

import argparse
import os
import pickle
import pandas as pd
from sentence_transformers import SentenceTransformer
import anthropic

from config import EMBEDDING_MODEL_NAME, THEMES
from scrape import get_app_metadata, get_reviews, get_competitor_ids
from preprocess import filter_language, filter_reviews, to_text_list, build_stopwords
from model import run_topic_model
from themes import build_topic_table

from sklearn.feature_extraction.text import CountVectorizer


def analyse_app(
    app_id: str,
    embedding_model: SentenceTransformer,
    client: anthropic.Anthropic,
    themes: list[str] = THEMES,
    output_dir: str = "data/outputs",
    verbose: bool = True,
) -> dict:
    """
    Full pipeline: app_id in → topic_table, theme_table, review_df out.

    Args:
        app_id:          Google Play app ID, e.g. 'com.duolingo'
        embedding_model: Pre-loaded SentenceTransformer instance (shared across apps)
        client:          Anthropic client instance
        themes:          List of theme names for assignment
        output_dir:      Directory to save parquet outputs
        verbose:         Print progress messages

    Returns:
        dict with keys: app_id, app_name, topic_table, theme_table, review_df,
                        topic_model, metadata
    """
    # ------------------------------------------------------------------
    # 1. Scrape
    # ------------------------------------------------------------------
    if verbose:
        print(f"\n{'='*50}")
        print(f"Analysing: {app_id}")
        print(f"{'='*50}")
        print("Scraping app metadata and reviews...")

    metadata = get_app_metadata(app_id)
    review_df = get_reviews(app_id)

    if verbose:
        print(f"  {len(review_df)} reviews fetched")

    # ------------------------------------------------------------------
    # 2. Preprocess
    # ------------------------------------------------------------------
    if verbose:
        print("Filtering reviews...")

    review_df = filter_language(review_df)
    review_df_filtered = filter_reviews(review_df)
    review_texts = to_text_list(review_df_filtered)

    if verbose:
        print(f"  {len(review_texts)} reviews after filtering")

    # ------------------------------------------------------------------
    # 3. Stopwords + vectorizer
    # ------------------------------------------------------------------
    custom_stopwords = build_stopwords(metadata)
    vectorizer = CountVectorizer(
        stop_words=custom_stopwords,
        ngram_range=(1, 2),
        min_df=2,
    )

    # ------------------------------------------------------------------
    # 4. Encode
    # ------------------------------------------------------------------
    if verbose:
        print("Encoding embeddings...")

    embeddings = embedding_model.encode(review_texts, show_progress_bar=verbose)

    # ------------------------------------------------------------------
    # 5. Topic modelling
    # ------------------------------------------------------------------
    if verbose:
        print("Running topic model...")

    topic_model, final_topics = run_topic_model(
        review_texts, embeddings, vectorizer, embedding_model, client, verbose=verbose
    )

    # ------------------------------------------------------------------
    # 6. Theme assignment + table construction
    # ------------------------------------------------------------------
    if verbose:
        print("Assigning themes and building tables...")

    topic_table, theme_table, review_df_out = build_topic_table(
        topic_model, review_df_filtered, client, themes
    )

    # ------------------------------------------------------------------
    # 7. Save outputs
    # ------------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    safe_id = app_id.replace(".", "_")

    topic_table.to_parquet(f"{output_dir}/{safe_id}_topics.parquet")
    theme_table.to_parquet(f"{output_dir}/{safe_id}_themes.parquet")
    review_df_out.to_parquet(f"{output_dir}/{safe_id}_reviews.parquet")

    # Save topic model separately (can't parquet a BERTopic object)
    with open(f"{output_dir}/{safe_id}_topic_model.pkl", "wb") as f:
        pickle.dump(topic_model, f)

    if verbose:
        print(f"\nOutputs saved to {output_dir}/")
        print(f"  {len(topic_table)} topics")
        print(f"  {len(theme_table)} themes")
        print(f"  {len(review_df_out)} reviews")

    return {
        "app_id": app_id,
        "app_name": metadata.get("title", app_id),
        "metadata": metadata,
        "topic_table": topic_table,
        "theme_table": theme_table,
        "review_df": review_df_out,
        "topic_model": topic_model,
    }


def run_competitor_analysis(
    app_id: str,
    embedding_model: SentenceTransformer,
    client: anthropic.Anthropic,
    output_dir: str = "data/outputs",
    verbose: bool = True,
) -> dict:
    """
    Run analyse_app on the main app and its competitors.
    Returns a dict keyed by app_id.
    """
    metadata = get_app_metadata(app_id)
    competitor_ids = get_competitor_ids(metadata)

    if verbose:
        print(f"Main app: {app_id}")
        print(f"Competitors: {competitor_ids}")

    all_results = {}
    for aid in [app_id] + competitor_ids:
        try:
            all_results[aid] = analyse_app(
                aid, embedding_model, client,
                output_dir=output_dir, verbose=verbose
            )
        except Exception as e:
            print(f"Failed to analyse {aid}: {e}")
            continue

    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AcquireIQ topic pipeline")
    parser.add_argument("--app_id", required=True, help="Google Play app ID")
    parser.add_argument("--competitors", action="store_true",
                        help="Also run on competitor apps")
    parser.add_argument("--output_dir", default="data/outputs",
                        help="Directory to save outputs")
    parser.add_argument("--hf_token", default=None,
                        help="HuggingFace token (optional for public models)")
    args = parser.parse_args()

    # Shared embedding model — load once
    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL_NAME,
        token=args.hf_token,
    )

    # Anthropic client — reads ANTHROPIC_API_KEY from environment
    client = anthropic.Anthropic()

    if args.competitors:
        run_competitor_analysis(
            args.app_id, embedding_model, client,
            output_dir=args.output_dir,
        )
    else:
        analyse_app(
            args.app_id, embedding_model, client,
            output_dir=args.output_dir,
        )
