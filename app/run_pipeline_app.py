import os
import streamlit as st

from data_helpers import OUTPUT_DIR

def run_pipeline(app_id: str, progress_placeholder, status_placeholder) -> bool:
    """
    Run the full analyse_app pipeline for a custom app.
    Updates progress_placeholder and status_placeholder as it goes.
    Returns True on success, False on failure.
    """
    # Import here so the heavy libraries don't slow down the initial page load
    try:
        from sentence_transformers import SentenceTransformer
        import anthropic
        from pipeline.pipeline import run_competitor_analysis
        from pipeline.config import EMBEDDING_MODEL_NAME
    except ImportError as e:
        status_placeholder.error(f"Import error: {e}")
        return False

    steps = [
        "Scraping reviews from Google Play...",
        "Filtering and encoding reviews...",
        "Fitting topic model (this takes a few minutes)...",
        "Labelling topics with Claude...",
        "Merging similar topics...",
        "Assigning themes...",
        "Saving outputs...",
    ]
    n_steps = len(steps)

    # We can't hook into the pipeline's internals for per-step progress,
    # so we show a pulsing status message and update progress at key checkpoints.
    # A future improvement would be to make analyse_app accept a callback.

    status_placeholder.info(f"Starting pipeline for `{app_id}`...")
    progress_placeholder.progress(0)

    try:
        api_key = st.secrets.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            status_placeholder.error(
                "ANTHROPIC_API_KEY not found. "
                "Add it to Streamlit secrets or set as environment variable."
            )
            return False

        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        client = anthropic.Anthropic(api_key=api_key)

        status_placeholder.info("⏳ Scraping and filtering reviews...")
        progress_placeholder.progress(1 / n_steps)

        # Run the full pipeline — this blocks until complete
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        run_competitor_analysis(
            app_id,
            embedding_model,
            client,
            output_dir=OUTPUT_DIR,
            verbose=False,
        )

        progress_placeholder.progress(1.0)
        status_placeholder.success(f"✅ Analysis complete for `{app_id}` and competitors")
        return True

    except Exception as e:
        status_placeholder.error(f"Pipeline failed: {e}")
        progress_placeholder.empty()
        return False
