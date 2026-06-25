THEMES = THEMES = [
    "Monetization",
    "Content Quality",
    "Technical Issues",
    "Feature Requests",
    "User Experience",
    "Account Issues"
]

review_prompt = """I have a cluster of Google Play app reviews containing the following documents:
    [DOCUMENTS]
    The cluster is described by these keywords: [KEYWORDS]

    Extract a short, specific label (max 6 words) naming the concrete issue, feature, or sentiment these reviews share. Avoid vague labels like "general feedback". Format:
    topic: <label>
    """
