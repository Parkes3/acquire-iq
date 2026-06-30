"""
run_pipeline.py
---------------
CLI entry point for the AcquireIQ pipeline.

Usage:
    python run_pipeline.py --app_id com.duolingo
    python run_pipeline.py --app_id com.duolingo --competitors
    python run_pipeline.py --app_id com.duolingo --output_dir my_outputs
"""

import argparse
from sentence_transformers import SentenceTransformer
import anthropic

from pipeline.config import EMBEDDING_MODEL_NAME
from pipeline.pipeline import analyse_app, run_competitor_analysis


def main():
    parser = argparse.ArgumentParser(description="AcquireIQ topic pipeline")
    parser.add_argument("--app_id", required=True, help="Google Play app ID e.g. com.duolingo")
    parser.add_argument("--competitors", action="store_true", help="Also run on competitor apps")
    parser.add_argument("--output_dir", default="data/outputs", help="Directory to save outputs")
    args = parser.parse_args()

    # Load once — shared across main app and any competitors
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from environment

    if args.competitors:
        run_competitor_analysis(
            args.app_id,
            embedding_model,
            client,
            output_dir=args.output_dir,
        )
    else:
        analyse_app(
            args.app_id,
            embedding_model,
            client,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()