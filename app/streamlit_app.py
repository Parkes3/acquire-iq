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
from PIL import Image
from pathlib import Path

from data_helpers import OUTPUT_DIR, load_app_data, load_groups, get_apps_in_group
from styling import THEME_COLORS, HEATMAP_PALETTE
from run_pipeline_app import run_pipeline
from plotting_and_tables import plot_pareto, plot_theme_bar, process_theme_table, process_topic_table

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

PARETO_CUTOFF = 0.8

# ---------------------------------------------------------------------------
# App Descrtiption to display
# ---------------------------------------------------------------------------

APP_DESCRIPTION = """**AcquireIQ** analyses negative reviews from Google Play to surface the issues that matter most to users.

Select a main app from the sidebar to explore its results across Themes, Topics and Review Explorer, you can then select any app from the competitor group to drill on details:

- **Overview** — theme-level breakdown of negative reviews, showing which broad categories (Monetization, Technical Issues, etc.) account for the most complaints. Includes the top upvoted reviews per theme.

- **Topic Detail** — a Pareto chart of individual topics ordered by severity, combining review volume, thumbs up count, and low rating into a single signal. Use the cutoff slider to focus on the issues that drive the most negative sentiment. Drill into any topic to read its most upvoted reviews.

- **Review Explorer** — filter individual reviews by theme and topic. Useful for reading the exact language users use when describing a problem.

- **Competitor View** — compare theme distributions across the full competitor group. Shows install counts and the share of negative reviews by theme for each app, making it easy to spot where your app over- or under-indexes relative to competitors.

**WORK IN PROGRESS:**
You can run your own analysis by entering a Google Play appId
    """


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
# Sidebar — app selection
# ---------------------------------------------------------------------------

with st.sidebar:
    st.image(Path(__file__).parent/'static'/'favicon-32x32.png')
    st.title("AcquireIQ")
    st.caption("Automated competitive intelligence platform for consumer mobile apps.")
    st.markdown("---")

    groups = load_groups(OUTPUT_DIR)
    active_group = None
    group_apps = pd.DataFrame()
    active_app_id = None
    app_choice = "- select -"

    # Check if a new app was just analysed and should be pre-selected
    pending = st.session_state.pop("pending_app_id", None)

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

        # If a new app was just run, find its index to pre-select it
        default_idx = 0
        if pending:
            st.caption(f"debug: pending={pending}")
            for i, (title, aid) in enumerate(main_app_options):
                st.caption(f"debug: checking {aid}")
                if aid == pending:
                    default_idx = i + 1  # +1 because "— select —" is index 0
                    break


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
            
            if active_app_id:
                sid = active_app_id.replace(".", "_")
                meta_path = f"{OUTPUT_DIR}/{sid}_metadata.parquet"
                if os.path.exists(meta_path):
                    meta = pd.read_parquet(meta_path).iloc[0]
                    icon_url = meta.get("Icon")
                    if icon_url:
                        st.image(icon_url, width=64)

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
            "⏳ This typically takes **15-20 minutes** depending on the number of reviews and number of competitors. "
            "Do not close this tab."
        )
        progress_bar = st.progress(0)
        status_msg = st.empty()  # placeholder we can update as pipeline runs

        success = run_pipeline(entered_id, progress_bar, status_msg)

        if success:

            # Clear all relevant caches
            load_app_data.clear()
            load_groups.clear()
            get_apps_in_group.clear()

            # Set session state so the new app is pre-selected after rerun
            st.session_state["pending_app_id"] = entered_id
            st.rerun()

# ---------------------------------------------------------------------------
# Main dashboard
# ---------------------------------------------------------------------------

if active_app_id is None:
    with main_area:
        st.title("AcquireIQ")
        st.markdown(APP_DESCRIPTION)
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


col_icon, col_title = st.columns([1, 8])
with col_icon:
    if icon_url:
        st.image(icon_url, width=60)
with col_title:
    st.title(f"{app_display_name}")
    st.caption(f"`{active_app_id}`")

st.caption(
    "Negative reviews analysed by topic and theme. "
    "Use the tabs below to explore complaints by category, drill into individual topics, "
    "browse raw reviews, or compare against competitors."
)

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

    display_theme = process_theme_table(theme_table, topic_table)

    display_theme = display_theme.sort_values(by='percentReviews', ascending=False)
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

    display_topics = process_topic_table(topic_table, theme_filter)
    st.dataframe(
        display_topics[[
            "Topic", "Theme", "numReviews", "numThumbsUp", "avgRating", 'Impact', "Severity"
        ]].reset_index(drop=True),
        use_container_width=True,
    )

    st.markdown("---")
    st.subheader("Top reviews by topic")
    st.caption("Most upvoted reviews for each topic. Filtered by theme selection above.")

    topics_to_show = display_topics.reset_index()
    for _, row in topics_to_show.iterrows():
        topic_label = row["Topic"]
        n_reviews = int(row["numReviews"])
        theme = row.get("Theme", "")
        colour = THEME_COLORS.get(theme, "#95A5A6")
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
    # Row of competitor icons above the installs chart
    icon_cols = st.columns(len(group_apps))
    for col, (_, app_row) in zip(icon_cols, group_apps.iterrows()):
        sid = app_row["app_id"].replace(".", "_")
        meta_path = f"{OUTPUT_DIR}/{sid}_metadata.parquet"
        if os.path.exists(meta_path):
            meta = pd.read_parquet(meta_path).iloc[0]
            icon_url = meta.get("Icon")
            with col:
                if icon_url:
                    st.image(icon_url, width=48)
                st.caption(app_row["Title"])
    st.subheader("Installs")
    fig, ax = plt.subplots(figsize=(8, 4))
# In competitor tab, replace is_main reference with active_app_id
    colours = [
        "#E74C3C" if row["app_id"] == active_app_id else "#95A5A6"
        for _, row in group_apps.iterrows()
    ]
    group_apps = group_apps.sort_values(by='NumInstalls', ascending=False)
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
        selected_title = group_apps.loc[
            group_apps["app_id"] == active_app_id, "Title"
        ].values[0]
        other_titles = [t for t in pivot.index if t != selected_title]
        if selected_title in pivot.index:
            pivot = pivot.loc[[selected_title] + other_titles]

        fig2, ax2 = plt.subplots(figsize=(10, 5))
        pivot.plot(
            kind="bar", stacked=False, ax=ax2,
            color=[THEME_COLORS.get(c, "#95A5A6") for c in pivot.columns],
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
            styler.background_gradient(vmin=0, vmax=100, cmap=HEATMAP_PALETTE)
            styler.format('{:.2f}%')
            return styler
        
        def highlight_main_app(row):
            if row.name == selected_title:
                return ['font-weight: bold'] * len(row)
            return [''] * len(row)
        
        def highlight_index(idx):
            if idx == selected_title:
                return 'font-weight: bold'
            return ''
        
        pivot_disp = (pivot*100).round(2)
        #tdf_pivot = tdf.pivot(index='App', columns='Theme', values='percentReviews')
        st.dataframe(pivot_disp
                     #.reset_index()
                     .style.pipe(background_styling)
                     .apply(highlight_main_app, axis=1)
                     .map_index(highlight_index, axis=0)
                     , use_container_width=True)


        st.markdown('---')
        st.subheader('Gap to Competitor Average')
        st.caption(f'Comparing {selected_title} to competitors based on share of reviews by theme')

        competitors = pivot_disp.loc[pivot_disp.index != selected_title].mean()
        competitors.name = 'Competitor Average'
        comparison_frame = pd.concat([pivot_disp.loc[selected_title], competitors], axis=1)
        comparison_frame['Difference'] = comparison_frame[selected_title] - comparison_frame['Competitor Average']
        comparison_frame = comparison_frame.sort_values(by='Difference', ascending=False)
        
        #plotting bar
        colors = ['crimson' if x > 0 else 'forestgreen' for x in comparison_frame.Difference]
        fig3, ax = plt.subplots(figsize=(6,4))
        ax.barh(comparison_frame.index, comparison_frame.Difference, color=colors)
        ax.xaxis.set_major_formatter(mtick.PercentFormatter())
        ax.set_xlabel('Percent Difference')
        st.pyplot(fig3, use_container_width=False)

        st.markdown("---")
        st.subheader("Metadata")

        display_meta = group_apps[[
            "Title", "NumInstalls", 'NumReviews', "Score", "AdSupported", 'DateReleased', #"is_main"
        ]].copy()
    
        display_meta['Score'] = display_meta['Score'].apply(lambda x: f"{x:,.2f}")
        display_meta['NumInstalls'] = display_meta['NumInstalls'].apply(lambda x: f"{x:,.0f}")
        display_meta['NumReviews'] = display_meta['NumReviews'].apply(lambda x: f"{x:,.0f}")

 
        display_meta = display_meta.set_index('Title')
        #we can add this ⭐
        #display_meta = display_meta.rename(columns={"is_main": "Main App"})
        st.dataframe(display_meta.style.apply(highlight_main_app, axis=1), use_container_width=True)