import pandas as pd

THEME_COLORS = {
    "Monetization":    "#E74C3C",
    "Technical Issues": "#E67E22",
    "Content Quality":  "#3498DB",
    "User Experience":  "#9B59B6",
    "Feature Requests": "#1ABC9C",
    "Account Issues":   "#F72CC4",
}

APP_DESCRIPTION = """
**AcquireIQ** analyses negative reviews from Google Play to surface the issues that matter most to users.

Select a main app from the sidebar to explore its results across Themes, Topics and Review Explorer, you can then select any app from the competitor group to drill on details:

- **Overview** — theme-level breakdown of negative reviews, showing which broad categories (Monetization, Technical Issues, etc.) account for the most complaints. Includes the top upvoted reviews per theme.

- **Topic Detail** — a Pareto chart of individual topics ordered by severity, combining review volume, thumbs up count, and low rating into a single signal. Use the cutoff slider to focus on the issues that drive the most negative sentiment. Drill into any topic to read its most upvoted reviews.

- **Review Explorer** — filter individual reviews by theme and topic. Useful for reading the exact language users use when describing a problem.

- **Competitor View** — compare theme distributions across the full competitor group. Shows install counts and the share of negative reviews by theme for each app, making it easy to spot where your app over- or under-indexes relative to competitors.

**WORK IN PROGRESS:**
You can run your own analysis by entering a Google Play appId
"""

HEATMAP_PALETTE = 'Reds'