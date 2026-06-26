"""
streamlit_app.py
----------------
AcquireIQ — interactive review topic explorer.
Loads pre-computed parquet outputs from data/outputs/.

Run with:
    streamlit run app/streamlit_app.py
"""

import os
import glob
import pickle
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AcquireIQ",
    page_icon="📊",
    layout="wide",
)

st.title("AcquireIQ — App Review Topic Explorer")

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

OUTPUT_DIR = "data/outputs"
PARETO_CUTOFF = 0.8

THEME_COLOURS = {
    "Monetization": "#E74C3C",
    "Technical Issues": "#E67E22",
    "Content Quality": "#3498DB",
    "User Experience": "#9B59B6",
    "Feature Requests": "#1ABC9C",
    "Account Issues": "#F39C12",
}


@st.cache_data
def load_app_data(app_id: str) -> dict:
    safe_id = app_id.replace(".", "_")
    topic_table = pd.read_parquet(f"{OUTPUT_DIR}/{safe_id}_topics.parquet")
    theme_table = pd.read_parquet(f"{OUTPUT_DIR}/{safe_id}_themes.parquet")
    review_df = pd.read_parquet(f"{OUTPUT_DIR}/{safe_id}_reviews.parquet")
    return {
        "topic_table": topic_table,
        "theme_table": theme_table,
        "review_df": review_df,
    }


def get_available_apps() -> list[str]:
    files = glob.glob(f"{OUTPUT_DIR}/*_topics.parquet")
    return [
        os.path.basename(f).replace("_topics.parquet", "").replace("_", ".")
        for f in files
    ]


available_apps = get_available_apps()

if not available_apps:
    st.warning(
        "No pre-computed results found in `data/outputs/`. "
        "Run `python pipeline/pipeline.py --app_id <id>` first."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar — app selector
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("App Selection")
    selected_app = st.selectbox("Select app", available_apps)
    data = load_app_data(selected_app)

    st.markdown("---")
    st.caption(
        f"{len(data['topic_table'])} topics · "
        f"{len(data['review_df'])} reviews"
    )

topic_table = data["topic_table"]
theme_table = data["theme_table"]
review_df = data["review_df"]

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab1, tab2, tab3 = st.tabs(["📊 Overview", "🔍 Topic Detail", "💬 Review Explorer"])


# ---------------------------------------------------------------------------
# Tab 1 — Overview
# ---------------------------------------------------------------------------

with tab1:
    st.subheader("Theme Summary")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total negative reviews", len(review_df[review_df["Topic"] != -1]))
    col2.metric("Topics found", len(topic_table))
    col3.metric("Themes", len(theme_table))

    st.markdown("---")

    # Theme bar chart
    fig, ax = plt.subplots(figsize=(8, 4))
    colours = [THEME_COLOURS.get(t, "#95A5A6") for t in theme_table.index]
    bars = ax.barh(theme_table.index, theme_table["numReviews"], color=colours)
    ax.set_xlabel("Number of reviews")
    ax.set_title("Negative reviews by theme")
    ax.bar_label(bars, fmt="%d", padding=3, fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("Theme Table")
    display_theme = theme_table[
        ["numTopics", "numReviews", "numThumbsUp", "avgRating",
         "percentReviews", "percentThumbsUp"]
    ].copy()
    display_theme["percentReviews"] = display_theme["percentReviews"].map("{:.1%}".format)
    display_theme["percentThumbsUp"] = display_theme["percentThumbsUp"].map("{:.1%}".format)
    display_theme["avgRating"] = display_theme["avgRating"].map("{:.2f}".format)
    st.dataframe(display_theme, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 2 — Topic Detail (Pareto chart)
# ---------------------------------------------------------------------------

with tab2:
    st.subheader("Topic Pareto Chart")

    cutoff_pct = st.slider(
        "Pareto cutoff", min_value=0.5, max_value=1.0,
        value=PARETO_CUTOFF, step=0.05, format="%.0f%%"
    )

    top_topics = topic_table[topic_table["CumulativePercentage"] <= cutoff_pct]

    if top_topics.empty:
        st.info("No topics below selected cutoff.")
    else:
        fig, ax1 = plt.subplots(figsize=(10, 5))

        sns.barplot(
            data=top_topics.reset_index(),
            x="Claude", y="thumbsAndReviews",
            hue="Theme",
            palette=THEME_COLOURS,
            ax=ax1,
        )
        ax1.set_ylabel("Reviews + Thumbs Up (weighted)")
        ax1.set_xlabel("")
        plt.xticks(rotation=30, ha="right", fontsize=7)

        ax2 = ax1.twinx()
        top_topics.reset_index().plot.line(
            x="Claude", y="CumulativePercentage",
            linestyle="--", ax=ax2, color="black", label="Cumulative %",
        )
        ax2.set_ylim(0, 1)
        ax2.set_ylabel("Cumulative %")
        ax2.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))

        separator = Line2D([0], [0], color="none")
        bar_handles, bar_labels = ax1.get_legend_handles_labels()
        line_handles, line_labels = ax2.get_legend_handles_labels()
        ax1.legend(
            handles=[separator] + bar_handles + [separator] + line_handles,
            labels=["Theme"] + bar_labels + [""] + line_labels,
            loc="upper left", fontsize=7,
        )
        if ax2.get_legend():
            ax2.get_legend().remove()

        plt.title(f"Pareto Chart — top topics ({cutoff_pct:.0%} cutoff)")
        plt.tight_layout()
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("Full Topic Table")

    theme_filter = st.multiselect(
        "Filter by theme",
        options=topic_table["Theme"].dropna().unique().tolist(),
        default=[],
    )

    display_topics = topic_table.copy()
    if theme_filter:
        display_topics = display_topics[display_topics["Theme"].isin(theme_filter)]

    st.dataframe(
        display_topics[["Claude", "Theme", "numReviews", "numThumbsUp",
                         "avgRating", "Severity"]].reset_index(drop=True),
        use_container_width=True,
    )


# ---------------------------------------------------------------------------
# Tab 3 — Review Explorer
# ---------------------------------------------------------------------------

with tab3:
    st.subheader("Review Explorer")

    col1, col2 = st.columns(2)

    with col1:
        theme_select = st.selectbox(
            "Theme",
            options=["All"] + sorted(review_df["Theme"].dropna().unique().tolist()),
        )
    with col2:
        topic_options = (
            review_df[review_df["Theme"] == theme_select]["ClaudeLabel"]
            .dropna().unique().tolist()
            if theme_select != "All"
            else review_df["ClaudeLabel"].dropna().unique().tolist()
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

    display_cols = ["content", "score", "thumbsUpCount", "ClaudeLabel", "Theme"]
    st.dataframe(
        filtered[display_cols]
        .sort_values("thumbsUpCount", ascending=False)
        .reset_index(drop=True),
        use_container_width=True,
    )
