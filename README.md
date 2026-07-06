# AcquireIQ

Competitive intelligence platform for consumer mobile apps utilising an automated Google Play review topic modelling pipeline.
Input an app ID, get structured topic and theme analysis of negative reviews.

## Project structure

```
acquireiq/
├── pipeline/
│   ├── config.py       # constants — themes, model names, thresholds
│   ├── scrape.py       # google-play-scraper wrappers
│   ├── preprocess.py   # filtering, language detection, stopwords
│   ├── model.py        # BERTopic fit, outlier reduction, merging, labelling
│   ├── themes.py       # Claude theme assignment, topic/theme table construction
│   └── pipeline.py     # analyse_app() + CLI entry point
├── app/
│   └── streamlit_app.py
├── data/
│   └── outputs/        # parquet files written here by pipeline.py
├── notebooks/
│   └── development.ipynb  # exploratory notebook showing pipeline development
├── run_pipeline.py
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm

export ANTHROPIC_API_KEY="sk-ant-..."
```

## Running the pipeline

```bash
# Single app
python pipeline/pipeline.py --app_id com.duolingo

# Main app + competitors
python pipeline/pipeline.py --app_id com.duolingo --competitors

# Custom output directory
python pipeline/pipeline.py --app_id com.duolingo --output_dir my_outputs
```

Outputs saved to `data/outputs/`:
- `{app_id}_topics.parquet`  — topic-level table
- `{app_id}_themes.parquet`  — theme-level table
- `{app_id}_reviews.parquet` — review-level table with topic/theme columns
- `{app_id}_topic_model.pkl` — fitted BERTopic object

## Running the Streamlit app

```bash
streamlit run app/streamlit_app.py
```

Loads all parquet files from `data/outputs/` automatically.

## Deleting apps:

To remove data for an app
```bash
# See all apps currently in outputs
python delete_app.py --list

# Delete a specific app (files + group entry)
python delete_app.py --app_id com.duolingo

# Remove from groups.json only, keep parquet files
python delete_app.py --app_id com.duolingo --group_only

# Nuclear option — wipe everything
python delete_app.py --all
```
## Configuration

Edit `pipeline/config.py` to change:
- `THEMES` — the theme categories for assignment
- `GENERIC_APP_WORDS` — words to always stopword
- `GENRE_ACTIVITY_WORDS` — genre-specific activity words to stopword
- `MIN_WORD_COUNT` / `MAX_SCORE` — review filter thresholds
- `TARGET_MIN_TOPICS` / `TARGET_MAX_TOPICS` — BERTopic fitting targets
- `EMBEDDING_MODEL_NAME` / `CLAUDE_MODEL` — model choices

## How it works

1. **Scrape** — fetches up to 3,000 English reviews with score ≤ 3
2. **Stopwords** — builds app-specific stopwords from title, genre, and generic app vocabulary
3. **Embed** — encodes reviews with `all-MiniLM-L6-v2`
4. **Fit BERTopic** — iteratively tunes HDBSCAN parameters until a stable topic count is found
5. **Reduce outliers** — reassigns unclustered reviews to nearest topic by c-TF-IDF
6. **Label** — applies KeyBERT + Claude representation models to generate topic labels
7. **Merge** — agglomerative clustering on combined Claude label + KeyBERT embeddings merges similar topics
8. **Theme assignment** — Claude assigns each topic to one of the predefined themes
9. **Output** — saves topic, theme, and review tables as parquet files
