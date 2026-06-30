"""
config.py
---------
Central configuration for AcquireIQ pipeline.
Edit THEMES and GENERIC_APP_WORDS here rather than touching pipeline code.
"""

THEMES = [
    "Monetization",
    "Content Quality",
    "Technical Issues",
    "Feature Requests",
    "User Experience",
    "Account Issues",
]

GENERIC_APP_WORDS = [
    "app", "apps", "play", "game", "games", "really", "just", "also",
    "use", "using", "used", "get", "got", "im", "ive", "dont", "doesnt",
    "install",
]

# Maps Google Play genreId values to activity verbs implied by that genre.
# These words appear generically across all reviews for that genre and
# carry no topic-discriminating signal.
GENRE_ACTIVITY_WORDS = {
    "EDUCATION": ["learn", "learning", "study", "studying", "lesson", "lessons"],
    "HEALTH_AND_FITNESS": ["workout", "workouts", "exercise", "train", "training"],
    "MUSIC_AND_AUDIO": ["listen", "listening", "play", "playing"],
    "DATING": ["date"],
}

# Review filtering
MIN_WORD_COUNT = 10
MAX_SCORE = 3           # reviews with score <= this are treated as negative

# BERTopic fitting
TARGET_MIN_TOPICS = 25
TARGET_MAX_TOPICS = 50
OUTLIER_THRESHOLD = 0.18
MAX_FIT_ATTEMPTS = 6

# Agglomerative clustering
MERGE_THRESHOLD_RANGE = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
MERGE_TARGET_MIN = 15
MERGE_TARGET_MAX = 25

# Models
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CLAUDE_MODEL = "claude-haiku-4-5-20251001"

# Review scraping
REVIEW_COUNT = 10000
REVIEW_LANG = "en"
REVIEW_COUNTRY = "us"
NUM_COMPETITORS = 6

# Pareto cutoff
PARETO_CUTOFF = 0.8
