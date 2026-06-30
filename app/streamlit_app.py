"""
streamlit_app.py
----------------
AcquireIQ — interactive review topic explorer.

Prebuilt apps load instantly from data/outputs/.
Custom apps run the full pipeline on demand (5+ minutes).

Run with:
    streamlit run app/streamlit_app.py
"""

import os
import sys
import glob
import time
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Path setup — so pipeline modules are importable from app/
# ---------------------------------------------------------------------------

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pipeline"))

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AcquireIQ",
    page_icon="📊",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "outputs")
PARETO_CUTOFF = 0.8

#TODO: change to get_available_apps func
PREBUILT_APPS = {
    "Duolingo":  "com.duolingo",
    "Calm":      "com.calm.android",
    "Boostcamp": "com.boostcamp.app",
    "Tinder":    "com.tinder",
    "Strava":    "com.strava",
}

THEME_COLOURS = {
    "Monetization":    "#E74C3C",
    "Technical Issues": "#E67E22",
    "Content Quality":  "#3498DB",
    "User Experience":  "#9B59B6",
    "Feature Requests": "#1ABC9C",
    "Account Issues":   "#F39C12",
}

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def safe_id(app_id: str) -> str:
    return app_id.replace(".", "_")


def outputs_exist(app_id: str) -> bool:
    return os.path.exists(f"{OUTPUT_DIR}/{safe_id(app_id)}_topics.parquet")


@st.cache_data
def load_app_data(app_id: str) -> dict:
    """Load pre-computed parquet files for an app. Cached so reruns are instant."""
    sid = safe_id(app_id)
    return {
        "topic_table": pd.read_parquet(f"{OUTPUT_DIR}/{sid}_topics.parquet"),
        "theme_table":  pd.read_parquet(f"{OUTPUT_DIR}/{sid}_themes.parquet"),
        "review_df":    pd.read_parquet(f"{OUTPUT_DIR}/{sid}_reviews.parquet"),
    }


def get_custom_app_ids() -> list[str]:
    """Return app IDs that exist in outputs but aren't in PREBUILT_APPS."""
    files = glob.glob(f"{OUTPUT_DIR}/*_topics.parquet")
    all_ids = set(
        os.path.basename(f).replace("_topics.parquet", "").replace("_", ".")
        for f in files
    )
    prebuilt_ids = set(PREBUILT_APPS.values())
    return sorted(all_ids - prebuilt_ids)


# ---------------------------------------------------------------------------
# Pipeline runner (only imported when needed — avoids slow imports on load)
# ---------------------------------------------------------------------------

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
        from pipeline.pipeline import analyse_app
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
        analyse_app(
            app_id,
            embedding_model,
            client,
            output_dir=OUTPUT_DIR,
            verbose=False,
        )

        progress_placeholder.progress(1.0)
        status_placeholder.success(f"✅ Pipeline complete for `{app_id}`")
        return True

    except Exception as e:
        status_placeholder.error(f"Pipeline failed: {e}")
        progress_placeholder.empty()
        return False


# ---------------------------------------------------------------------------
# Review display helper
# ---------------------------------------------------------------------------

def show_top_reviews(review_df: pd.DataFrame, filter_col: str, filter_val: str, n: int = 5):
    """
    Show top N reviews by thumbs up count for a given topic or theme.
    Renders each review as a styled card inside an expander.
    """
    subset = (
        review_df[
            (review_df[filter_col] == filter_val) &
            (review_df["Topic"] != -1)
        ]
        .sort_values("thumbsUpCount", ascending=False)
        .head(n)
    )

    if subset.empty:
        st.caption("No reviews found.")
        return

    for _, row in subset.iterrows():
        stars = "⭐" * int(row["score"])
        thumbs = int(row.get("thumbsUpCount", 0))
        st.markdown(
            f"""
            <div style="
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                padding: 12px 16px;
                margin-bottom: 8px;
                background: #fafafa;
            ">
                <div style="font-size:12px; color:#888; margin-bottom:4px">
                    {stars} &nbsp;·&nbsp; 👍 {thumbs:,}
                </div>
                <div style="font-size:13px; line-height:1.5">
                    {row['content']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_theme_bar(theme_table: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4))
    colours = [THEME_COLOURS.get(t, "#95A5A6") for t in theme_table.index]
    bars = ax.barh(theme_table.index, theme_table["numReviews"], color=colours)
    ax.set_xlabel("Number of reviews")
    ax.set_title("Negative reviews by theme")
    ax.bar_label(bars, fmt="%d", padding=3, fontsize=9)
    plt.tight_layout()
    return fig


def plot_pareto(topic_table: pd.DataFrame, cutoff: float):
    top = topic_table[topic_table["CumulativePercentage"] <= cutoff]
    if top.empty:
        return None

    fig, ax1 = plt.subplots(figsize=(12, 5))
    sns.barplot(
        data=top.reset_index(),
        x="Claude", y="thumbsAndReviews",
        hue="Theme", palette=THEME_COLOURS, ax=ax1,
    )
    ax1.set_ylabel("Reviews + Thumbs Up (weighted)")
    ax1.set_xlabel("")
    plt.xticks(rotation=30, ha="right", fontsize=7)

    ax2 = ax1.twinx()
    top.reset_index().plot.line(
        x="Claude", y="CumulativePercentage",
        linestyle="--", ax=ax2, color="black", label="Cumulative %",
    )
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Cumulative %")
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))

    sep = Line2D([0], [0], color="none")
    bh, bl = ax1.get_legend_handles_labels()
    lh, ll = ax2.get_legend_handles_labels()
    ax1.legend(
        handles=[sep] + bh + [sep] + lh,
        labels=["Theme"] + bl + [""] + ll,
        loc="upper left", fontsize=7,
    )
    if ax2.get_legend():
        ax2.get_legend().remove()

    plt.title(f"Pareto Chart — top topics ({cutoff:.0%} cumulative)")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Sidebar — app selection
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("AcquireIQ")
    st.caption("App review topic explorer")
    st.markdown("---")

    # --- Prebuilt apps ---
    st.subheader("Prebuilt apps")
    st.caption("Results load instantly.")

    available_prebuilt = {
        name: aid
        for name, aid in PREBUILT_APPS.items()
        if outputs_exist(aid)
    }
    missing_prebuilt = {
        name: aid
        for name, aid in PREBUILT_APPS.items()
        if not outputs_exist(aid)
    }

    if available_prebuilt:
        prebuilt_choice = st.selectbox(
            "Select an app",
            options=["— select —"] + list(available_prebuilt.keys()),
            key="prebuilt_select",
        )
    else:
        st.info("No prebuilt results found. Run the pipeline to generate them.")
        prebuilt_choice = "— select —"

    if missing_prebuilt:
        with st.expander("Not yet generated"):
            for name in missing_prebuilt:
                st.caption(f"• {name}")

    st.markdown("---")

    # --- Custom app ---
    st.subheader("Custom app")
    st.caption(
        "Enter a Google Play app ID (e.g. `com.spotify.music`). "
        "The pipeline will run — **this takes 5+ minutes**."
    )

    custom_input = st.text_input(
        "App ID",
        placeholder="com.example.app",
        key="custom_input",
    )

    # Show previously run custom apps if any exist
    custom_done = get_custom_app_ids()
    if custom_done:
        st.caption("Previously run custom apps:")
        custom_choice = st.selectbox(
            "Load custom result",
            options=["— select —"] + custom_done,
            key="custom_select",
        )
    else:
        custom_choice = "— select —"

    run_button = st.button(
        "▶ Run pipeline",
        disabled=not custom_input.strip(),
        use_container_width=True,
    )

    st.markdown("---")

# ---------------------------------------------------------------------------
# Resolve which app is active
# ---------------------------------------------------------------------------

# Determine active app_id from sidebar state
active_app_id = None

if prebuilt_choice != "— select —":
    active_app_id = available_prebuilt[prebuilt_choice]
elif custom_choice != "— select —":
    active_app_id = custom_choice

# ---------------------------------------------------------------------------
# Pipeline execution (custom app)
# ---------------------------------------------------------------------------

main_area = st.container()

if run_button and custom_input.strip():
    entered_id = custom_input.strip()

    with main_area:
        st.subheader(f"Running pipeline for `{entered_id}`")
        st.warning(
            "⏳ This typically takes **5–10 minutes** depending on the number of reviews. "
            "Do not close this tab."
        )
        progress_bar = st.progress(0)
        status_msg = st.empty()  # placeholder we can update as pipeline runs

        success = run_pipeline(entered_id, progress_bar, status_msg)

        if success:
            # Clear cache so new data loads
            load_app_data.clear()
            active_app_id = entered_id
            st.rerun()  # rerun so the new app appears in the dropdown and renders

# ---------------------------------------------------------------------------
# Main dashboard
# ---------------------------------------------------------------------------

if active_app_id is None:
    with main_area:
        st.title("AcquireIQ")
        st.markdown(
            "Select a **prebuilt app** from the sidebar to explore results instantly, "
            "or enter a **custom app ID** to run the full analysis pipeline."
        )
        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Prebuilt apps")
            for name, aid in PREBUILT_APPS.items():
                icon = "✅" if outputs_exist(aid) else "⏳"
                st.caption(f"{icon} {name} (`{aid}`)")
        with col2:
            st.subheader("How it works")
            st.markdown(
                """
                1. Reviews are scraped from Google Play (score ≤ 3)
                2. BERTopic clusters reviews into topics
                3. Claude labels each topic
                4. Similar topics are merged
                5. Topics are assigned to themes
                """
            )
    st.stop()

# Load data for the active app
try:
    data = load_app_data(active_app_id)
except FileNotFoundError:
    st.error(f"No data found for `{active_app_id}`. Run the pipeline first.")
    st.stop()

topic_table = data["topic_table"]
theme_table  = data["theme_table"]
review_df    = data["review_df"]

# App name for display
app_display_name = next(
    (name for name, aid in PREBUILT_APPS.items() if aid == active_app_id),
    active_app_id,
)

st.title(f"📊 {app_display_name}")
st.caption(f"`{active_app_id}`")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab1, tab2, tab3 = st.tabs(["Overview", "Topic Detail", "Review Explorer"])

# ---------------------------------------------------------------------------
# Tab 1 — Overview
# ---------------------------------------------------------------------------

with tab1:
    col1, col2, col3 = st.columns(3)
    col1.metric("Negative reviews analysed", len(review_df[review_df["Topic"] != -1]))
    col2.metric("Topics found",  len(topic_table))
    col3.metric("Themes",        len(theme_table))

    st.markdown("---")
    st.subheader("Reviews by theme")
    st.pyplot(plot_theme_bar(theme_table))

    st.markdown("---")
    st.subheader("Theme breakdown")

    display_theme = theme_table[[
        "numTopics", "numReviews", "numThumbsUp", "avgRating",
        "percentReviews", "percentThumbsUp",
    ]].copy()
    display_theme["percentReviews"]  = display_theme["percentReviews"].map("{:.1%}".format)
    display_theme["percentThumbsUp"] = display_theme["percentThumbsUp"].map("{:.1%}".format)
    display_theme["avgRating"]       = display_theme["avgRating"].map("{:.2f}".format)
    st.dataframe(display_theme, use_container_width=True)

    st.markdown("---")
    st.subheader("Top reviews by theme")
    st.caption("Most upvoted reviews for each theme.")

    for theme_name in theme_table.index:
        n_reviews = int(theme_table.loc[theme_name, "numReviews"])
        with st.expander(f"{theme_name}  ·  {n_reviews:,} reviews"):
            show_top_reviews(review_df, "Theme", theme_name)

# ---------------------------------------------------------------------------
# Tab 2 — Topic Detail
# ---------------------------------------------------------------------------

with tab2:
    st.subheader("Pareto chart")
    st.caption(
        "Topics are ordered by Severity (a weighted combination of review volume, "
        "thumbs up count, and low rating). The cumulative line shows what share of "
        "total signal is captured by the topics to its left."
    )

    cutoff_pct = st.slider(
        "Cumulative cutoff",
        min_value=0.5, max_value=1.0,
        value=PARETO_CUTOFF, step=0.05,
        format="%.0f%%",
    )

    fig = plot_pareto(topic_table, cutoff_pct)
    if fig:
        st.pyplot(fig)
    else:
        st.info("No topics below the selected cutoff.")

    st.markdown("---")
    st.subheader("All topics")

    theme_filter = st.multiselect(
        "Filter by theme",
        options=sorted(topic_table["Theme"].dropna().unique().tolist()),
    )
    display_topics = topic_table.copy()
    if theme_filter:
        display_topics = display_topics[display_topics["Theme"].isin(theme_filter)]

    st.dataframe(
        display_topics[[
            "Claude", "Theme", "numReviews", "numThumbsUp", "avgRating", "Severity"
        ]].reset_index(drop=True),
        use_container_width=True,
    )

    st.markdown("---")
    st.subheader("Top reviews by topic")
    st.caption("Most upvoted reviews for each topic. Filtered by theme selection above.")

    topics_to_show = display_topics.reset_index()
    for _, row in topics_to_show.iterrows():
        topic_label = row["Claude"]
        n_reviews = int(row["numReviews"])
        theme = row.get("Theme", "")
        colour = THEME_COLOURS.get(theme, "#95A5A6")
        with st.expander(f"{topic_label}  ·  {n_reviews:,} reviews  ·  {theme}"):
            show_top_reviews(review_df, "ClaudeLabel", topic_label)

# ---------------------------------------------------------------------------
# Tab 3 — Review Explorer
# ---------------------------------------------------------------------------

with tab3:
    st.subheader("Review Explorer")
    st.caption("Drill down into individual reviews by theme and topic.")

    col1, col2 = st.columns(2)

    with col1:
        theme_select = st.selectbox(
            "Theme",
            options=["All"] + sorted(review_df["Theme"].dropna().unique().tolist()),
        )
    with col2:
        if theme_select == "All":
            topic_options = review_df["ClaudeLabel"].dropna().unique().tolist()
        else:
            topic_options = (
                review_df[review_df["Theme"] == theme_select]["ClaudeLabel"]
                .dropna().unique().tolist()
            )
        topic_select = st.selectbox(
            "Topic", options=["All"] + sorted(topic_options)
        )

    filtered = review_df[review_df["Topic"] != -1].copy()
    if theme_select != "All":
        filtered = filtered[filtered["Theme"] == theme_select]
    if topic_select != "All":
        filtered = filtered[filtered["ClaudeLabel"] == topic_select]

    st.caption(f"{len(filtered)} reviews")
    st.dataframe(
        filtered[["content", "score", "thumbsUpCount", "ClaudeLabel", "Theme"]]
        .sort_values("thumbsUpCount", ascending=False)
        .reset_index(drop=True),
        use_container_width=True,
    )