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
import json
import time
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from matplotlib.lines import Line2D
from PIL import Image
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — so pipeline modules are importable from app/
# ---------------------------------------------------------------------------

# sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pipeline"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AcquireIQ",
    #page_icon="📒",
    page_icon=Image.open(Path(__file__).parent / 'static'/ 'aIQ_icon.ico'),
    layout="wide",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "outputs")
PARETO_CUTOFF = 0.8

#TODO: change to get_available_apps func
# DONE
# PREBUILT_APPS = {
#     "Duolingo":  "com.duolingo",
#     "Calm":      "com.calm.android",
#     "Boostcamp": "com.boostcamp.app",
#     "Tinder":    "com.tinder",
#     "Strava":    "com.strava",
# }

THEME_COLOURS = {
    "Monetization":    "#E74C3C",
    "Technical Issues": "#E67E22",
    "Content Quality":  "#3498DB",
    "User Experience":  "#9B59B6",
    "Feature Requests": "#1ABC9C",
    "Account Issues":   "#F72CC4",
}

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

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

    groups = load_groups(OUTPUT_DIR)
    active_group = None
    group_apps = pd.DataFrame()
    active_app_id = None
    app_choice = "- select -"
    if groups:
        st.subheader("Select main app")
        st.caption("Apps you have run the pipeline on.")

        # Build title → main_app_id mapping
        main_app_options = []
        for g in groups:
            sid = g["main_app_id"].replace(".", "_")
            meta_path = f"{OUTPUT_DIR}/{sid}_metadata.parquet"
            if os.path.exists(meta_path):
                meta = pd.read_parquet(meta_path).iloc[0]
                title = meta.get("Title", g["main_app_id"])
            else:
                title = g["main_app_id"]
            main_app_options.append((title, g["main_app_id"]))

        selected_title = st.selectbox(
            "Main app",
            options=["— select —"] + [t[0] for t in main_app_options]
        )

        if selected_title != "— select —":
            selected_main_app_id = next(
                aid for title, aid in main_app_options if title == selected_title
            )
            active_group = next(
                g for g in groups if g["main_app_id"] == selected_main_app_id
            )
            group_apps = get_apps_in_group(selected_main_app_id, OUTPUT_DIR)

            st.markdown("---")
            st.subheader("View app detail")
            st.caption("Select any app in this group to explore its topics and reviews.")

            app_choice = st.selectbox(
                "App",
                options=["— select —"] + group_apps["Title"].tolist()
            )
            if app_choice != "— select —":
                active_app_id = group_apps.loc[
                    group_apps["Title"] == app_choice, "app_id"
                ].values[0]

    else:
        st.info("No groups found. Run the pipeline with --competitors first.")

    st.markdown("---")
    st.subheader("Add a new app")
    st.caption("Enter a Google Play app ID. Running takes **5+ minutes**.")
    custom_input = st.text_input("App ID", placeholder="com.example.app")
    run_button = st.button(
        "▶ Run pipeline",
        disabled=not custom_input.strip(),
        use_container_width=True,
    )


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

# ---------------------------------------------------------------------------
# Main dashboard
# ---------------------------------------------------------------------------

if active_app_id is None:
    with main_area:
        st.title("AcquireIQ")
        st.markdown(
            "Select a **competitor group** and **app** from the sidebar to explore results, "
            "or enter a **custom app ID** to run the full analysis pipeline."
        )
        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Available groups")
            groups = load_groups(OUTPUT_DIR)
            if groups:
                for g in groups:
                    n_apps = len(g["app_ids"])
                    st.caption(f"**{g['main_app_id']}** — {n_apps} apps")
            else:
                st.caption("No groups found. Run the pipeline with --competitors first.")
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
app_display_name = group_apps.loc[
    group_apps['app_id'] == active_app_id, "Title"
    ].values[0] if active_app_id else active_app_id

st.title(f"📊 {app_display_name}")
st.caption(f"`{active_app_id}`")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Topic Detail", "Review Explorer", 'Competitor View'])

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

    cutoff_pct_int = st.slider(
        "Cumulative cutoff",
        min_value=50, max_value=100,
        value=int(PARETO_CUTOFF * 100), step=1,
        format="%d%%",
    )

    cutoff_pct = cutoff_pct_int / 100

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

with tab4:
    st.subheader("Competitor comparison")

    if active_group is None or group_apps.empty:
        st.info("Select a competitor group from the sidebar.")
        st.stop()

    if len(group_apps) < 2:
        st.info("Only one app in this group. Run with --competitors to add more.")
        st.stop()

    # --- Installs ---
    st.subheader("Installs")
    fig, ax = plt.subplots(figsize=(8, 4))
    colours = [
        "#E74C3C" if row["is_main"] else "#95A5A6"
        for _, row in group_apps.iterrows()
    ]
    bars = ax.barh(group_apps["Title"], group_apps["NumInstalls"], color=colours)
    ax.set_xlabel("Installs")
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(
            lambda x, _: f"{x/1e6:.0f}M" if x >= 1e6 else f"{x/1e3:.0f}K"
        )
    )
    ax.bar_label(
        bars,
        labels=[
            f"{r/1e6:.1f}M" if r >= 1e6 else f"{r/1e3:.0f}K"
            for r in group_apps["NumInstalls"]
        ],
        padding=3, fontsize=9,
    )
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")

    # --- Theme breakdown ---
    st.subheader("Theme breakdown")
    st.caption("Share of negative reviews per theme across the competitor group.")

    theme_rows = []
    for _, app_row in group_apps.iterrows():
        sid = app_row["app_id"].replace(".", "_")
        theme_path = f"{OUTPUT_DIR}/{sid}_themes.parquet"
        if not os.path.exists(theme_path):
            continue
        tdf = pd.read_parquet(theme_path).reset_index()
        tdf["App"] = app_row["Title"]
        theme_rows.append(tdf)

    if not theme_rows:
        st.info("No theme data found for this group yet.")
    else:
        all_themes = pd.concat(theme_rows, ignore_index=True)
        pivot = all_themes.pivot_table(
            index="App", columns="Theme",
            values="percentReviews", fill_value=0
        )

        # Main app first
        main_title = group_apps.loc[group_apps["is_main"], "Title"].values[0]
        other_titles = [t for t in pivot.index if t != main_title]
        if main_title in pivot.index:
            pivot = pivot.loc[[main_title] + other_titles]

        fig2, ax2 = plt.subplots(figsize=(10, 5))
        pivot.plot(
            kind="bar", stacked=False, ax=ax2,
            color=[THEME_COLOURS.get(c, "#95A5A6") for c in pivot.columns],
            width=0.75,
        )
        ax2.set_ylabel("% of negative reviews")
        ax2.set_xlabel("")
        ax2.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
        ax2.set_ylim(0,1)
        plt.xticks(rotation=20, ha="right")
        plt.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, title="Theme")
        plt.tight_layout()
        st.pyplot(fig2)

        st.markdown("---")
        st.subheader("Theme Dataframe")

        def background_styling(styler):
            styler.background_gradient(vmin=0, vmax=100, cmap='Reds')
            styler.format('{:.2f}%')
            return styler
        
        pivot_disp = (pivot*100).round(2)
        #tdf_pivot = tdf.pivot(index='App', columns='Theme', values='percentReviews')
        st.dataframe(pivot_disp
                     #.reset_index()
                     .style.pipe(background_styling)
                     , use_container_width=True)

        st.markdown("---")
        st.subheader("Metadata")
        display_meta = group_apps[[
            "Title", "NumInstalls", 'NumReviews', "Score", "AdSupported", 'DateReleased', #"is_main"
        ]].copy()
        display_meta["NumInstalls"] = display_meta["NumInstalls"].apply(lambda x: f"{x:,}")
        display_meta["NumReviews"] = display_meta["NumReviews"].apply(lambda x: f"{x:,}")
        
        display_meta["Score"] = (display_meta["Score"]
                                 .apply(lambda x: f"{x:.2f}")
                                 )
        #we can add this ⭐
        #display_meta = display_meta.rename(columns={"is_main": "Main App"})
        st.dataframe(display_meta.reset_index(drop=True), use_container_width=True)