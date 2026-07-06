import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

from styling import THEME_COLORS


def plot_theme_bar(theme_table: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4))
    colours = [THEME_COLORS.get(t, "#95A5A6") for t in theme_table.index]
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
        hue="Theme", palette=THEME_COLORS, ax=ax1,
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

def process_theme_table(theme_table, topic_table):
    display_theme = theme_table[["numThumbsUp", "avgRating",
    "percentReviews", "percentThumbsUp",
    ]].copy()
    display_theme["percentReviews"]  = display_theme["percentReviews"].map("{:.1%}".format)
    display_theme["percentThumbsUp"] = display_theme["percentThumbsUp"].map("{:.1%}".format)
    display_theme["avgRating"]       = display_theme["avgRating"].map("{:.2f}".format)

    top_topic_by_theme_idx = topic_table.groupby('Theme')['Severity'].idxmax()
    top_topic_by_theme = topic_table.loc[top_topic_by_theme_idx, ['Theme', 'Claude', 'Severity']]
    top_topic_by_theme = top_topic_by_theme.reset_index(drop=True).set_index('Theme')
    
    #Assiging impact label
    top_topic_by_theme = top_topic_by_theme.assign(Impact='🟢')
    top_topic_by_theme.loc[top_topic_by_theme['Severity'] > topic_table['Severity'].quantile(0.5), 'Impact'] = '🟠'
    top_topic_by_theme.loc[top_topic_by_theme['Severity'] > topic_table['Severity'].quantile(0.8), 'Impact'] = '🔴'
    
    top_topic_by_theme['High Priority Issue (Severity)'] = top_topic_by_theme.apply(lambda x: f'{x['Impact']} {x['Claude']}  ({x['Severity']:.2f})', axis=1)
    
    display_theme = display_theme.join(top_topic_by_theme['High Priority Issue (Severity)'])
    return display_theme

def process_topic_table(topic_table, theme_filter):
    display_topics = topic_table.copy()
    if theme_filter:
        display_topics = display_topics[display_topics["Theme"].isin(theme_filter)]

    display_topics = display_topics.rename(columns={'Claude': 'Topic'})
    
    display_topics = display_topics.assign(Impact='Low 🟢')
    display_topics.loc[display_topics['Severity'] > display_topics['Severity'].quantile(0.5), 'Impact'] = 'Medium 🟠'
    display_topics.loc[display_topics['Severity'] > display_topics['Severity'].quantile(0.8), 'Impact'] = 'High 🔴'
    return display_topics
    