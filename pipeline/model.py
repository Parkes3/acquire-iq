"""
model.py
--------
BERTopic fitting, outlier reduction, topic merging, and Claude/KeyBERT labelling.
"""

import math
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from bertopic.representation._base import BaseRepresentation
from bertopic.representation._utils import truncate_document

from pipeline.config import (
    TARGET_MIN_TOPICS,
    TARGET_MAX_TOPICS,
    OUTLIER_THRESHOLD,
    MAX_FIT_ATTEMPTS,
    MERGE_THRESHOLD_RANGE,
    MERGE_TARGET_MIN,
    MERGE_TARGET_MAX,
    CLAUDE_MODEL,
)


# ---------------------------------------------------------------------------
# Vectorizer safety
# ---------------------------------------------------------------------------

def min_safe_topic_size(min_df: int, max_df: float, buffer: int = 5) -> int:
    """
    Return the smallest cluster size at which CountVectorizer(min_df, max_df)
    cannot raise a ValueError due to max_df < min_df within a topic.
    """
    n = min_df
    while math.floor(max_df * n) < min_df:
        n += 1
    return n + buffer


# ---------------------------------------------------------------------------
# Claude representation model
# ---------------------------------------------------------------------------

REVIEW_PROMPT = """I have a cluster of Google Play app reviews containing the following documents:
[DOCUMENTS]
The cluster is described by these keywords: [KEYWORDS]

Extract a short, specific label (max 6 words) naming the concrete issue, feature, or sentiment these reviews share. Avoid vague labels like "general feedback". Format:
topic: <label>
"""


class ClaudeRepresentation(BaseRepresentation):
    def __init__(
        self,
        client,
        model: str = CLAUDE_MODEL,
        prompt: str = None,
        nr_docs: int = 4,
        doc_length=None,
        tokenizer=None,
    ):
        self.client = client
        self.model = model
        self.prompt = prompt or REVIEW_PROMPT
        self.nr_docs = nr_docs
        self.doc_length = doc_length
        self.tokenizer = tokenizer

    def extract_topics(self, topic_model, documents, c_tf_idf, topics):
        repr_docs_mappings, _, _, _ = topic_model._extract_representative_docs(
            c_tf_idf, documents, topics, nr_samples=500, nr_repr_docs=self.nr_docs
        )
        updated_topics = {}
        for topic, docs in repr_docs_mappings.items():
            keywords = ", ".join([w for w, _ in topics[topic]])
            doc_text = "\n".join(
                truncate_document(topic_model, self.doc_length, self.tokenizer, d)
                for d in docs
            )
            prompt = (
                self.prompt
                .replace("[DOCUMENTS]", doc_text)
                .replace("[KEYWORDS]", keywords)
            )
            response = self.client.messages.create(
                model=self.model,
                max_tokens=50,
                messages=[{"role": "user", "content": prompt}],
            )
            label = response.content[0].text.strip()
            if label.lower().startswith("topic:"):
                label = label.split(":", 1)[1].strip()
            updated_topics[topic] = [(label, 1)] + topics[topic][1:]
        return updated_topics


# ---------------------------------------------------------------------------
# BERTopic fitting
# ---------------------------------------------------------------------------

def _build_topic_model(
    embedding_model,
    vectorizer: CountVectorizer,
    min_cluster_size: int,
    min_samples: int,
    n_neighbors: int,
) -> BERTopic:
    umap_model = UMAP(
        n_neighbors=n_neighbors,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    return BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        min_topic_size=min_cluster_size,
        language="english",
        verbose=False,
    )


def fit_until_stable(
    review_texts: list[str],
    embeddings: np.ndarray,
    vectorizer: CountVectorizer,
    embedding_model,
    target_min: int = TARGET_MIN_TOPICS,
    target_max: int = TARGET_MAX_TOPICS,
    outlier_threshold: float = OUTLIER_THRESHOLD,
    max_attempts: int = MAX_FIT_ATTEMPTS,
    verbose: bool = True,
) -> tuple[BERTopic, list, np.ndarray]:
    """
    Iteratively fit BERTopic, adjusting min_cluster_size and min_samples
    until topic count and outlier rate fall within target ranges.
    Returns (topic_model, topics, probs).
    """
    n_docs = len(review_texts)
    min_cluster_size = max(10, n_docs // 150)
    min_samples = 7
    n_neighbors = 5
    prev_params = None

    # Safe defaults in case loop doesn't execute
    topic_model = None
    topics = []
    probs = np.array([])

    for attempt in range(max_attempts):
        current_params = (min_cluster_size, min_samples, n_neighbors)
        if current_params == prev_params:
            if verbose:
                print("Parameters unchanged — cannot improve further, stopping early")
            break
        prev_params = current_params

        if verbose:
            print(
                f"Attempt {attempt + 1}: min_cluster_size={min_cluster_size}, "
                f"min_samples={min_samples}, n_neighbors={n_neighbors}"
            )

        topic_model = _build_topic_model(
            embedding_model, vectorizer, min_cluster_size, min_samples, n_neighbors
        )
        topics, probs = topic_model.fit_transform(review_texts, embeddings=embeddings)

        info = topic_model.get_topic_info()
        n_real = len(info[info["Topic"] != -1])
        outlier_rows = info[info["Topic"] == -1]
        outlier_pct = (
            outlier_rows["Count"].values[0] / n_docs
            if len(outlier_rows) > 0
            else 0.0
        )

        if verbose:
            print(f"  → {n_real} topics, {outlier_pct:.1%} outliers")

        if target_min <= n_real <= target_max and outlier_pct < outlier_threshold:
            if verbose:
                print("Stable result found")
            return topic_model, topics, probs

        if n_real < target_min:
            min_cluster_size = max(5, int(min_cluster_size * 0.8))
            min_samples = max(3, min_cluster_size // 2)
        elif n_real > target_max:
            min_cluster_size = int(min_cluster_size * 1.2)
            min_samples = max(5, min_cluster_size // 2)
        elif outlier_pct >= outlier_threshold:
            min_samples = max(3, int(min_samples * 0.8))

    if topic_model is None:
        raise RuntimeError("fit_until_stable: no attempts were made — check max_attempts")
    if verbose:
        print("Warning: did not converge — returning last result")
    return topic_model, topics, probs


def reduce_outliers(
    topic_model: BERTopic,
    review_texts: list[str],
    topics: list,
    vectorizer: CountVectorizer,
) -> list:
    """Chain c-tf-idf outlier reduction and update model in place."""
    new_topics = topic_model.reduce_outliers(
        review_texts, topics, strategy="c-tf-idf"
    )
    topic_model.update_topics(
        review_texts, topics=new_topics, vectorizer_model=vectorizer
    )
    return new_topics


# ---------------------------------------------------------------------------
# Similarity matrix + agglomerative merging
# ---------------------------------------------------------------------------

def get_sim_matrix(info, embedding_model) -> np.ndarray:
    """
    Embed combined Claude label + KeyBERT keywords per topic
    and return a cosine similarity matrix.
    """
    info = info.copy()
    info["claude_label"] = info["Claude"].apply(
        lambda x: x[0][0] if isinstance(x[0], tuple) else str(x[0])
    )
    info["combined"] = info.apply(
        lambda row: " ".join([
            row["claude_label"],
            " ".join(
                item[0] if isinstance(item, tuple) else str(item)
                for item in row["KeyBERT"]
            ),
        ]),
        axis=1,
    )
    topic_embeddings = embedding_model.encode(
        info["combined"].tolist(), show_progress_bar=False
    )
    return cosine_similarity(topic_embeddings)


def select_threshold(
    sim_matrix: np.ndarray,
    info,
    thresholds: list[float] = MERGE_THRESHOLD_RANGE,
    target_min: int = MERGE_TARGET_MIN,
    target_max: int = MERGE_TARGET_MAX,
    verbose: bool = True,
) -> float:
    """
    Sweep distance thresholds and return the lowest one that keeps
    cluster count within [target_min, target_max].
    Falls back to the threshold closest to the midpoint of the target range.
    """
    results = []
    for threshold in thresholds:
        clusterer = AgglomerativeClustering(
            metric="precomputed",
            linkage="average",
            distance_threshold=threshold,
            n_clusters=None,
        )
        labels = clusterer.fit_predict(1 - sim_matrix)
        n_clusters = len(set(labels))
        n_merges = len(info) - n_clusters
        results.append((threshold, n_clusters, n_merges))
        if verbose:
            print(f"threshold={threshold} → {n_clusters} clusters, {n_merges} merges")

    for threshold, n_clusters, _ in results:
        if target_min <= n_clusters <= target_max:
            if verbose:
                print(f"\nSelected: threshold={threshold} → {n_clusters} clusters")
            return threshold

    target_mid = (target_min + target_max) / 2
    best = min(results, key=lambda x: abs(x[1] - target_mid))
    if verbose:
        print(f"\nFallback: threshold={best[0]} → {best[1]} clusters")
    return best[0]


def merge_similar_topics(
    topic_model: BERTopic,
    review_texts: list[str],
    info,
    embedding_model,
    verbose: bool = True,
) -> BERTopic:
    """
    Compute similarity matrix, select threshold, merge similar topics,
    return updated model.
    """
    sim_matrix = get_sim_matrix(info, embedding_model)
    threshold = select_threshold(sim_matrix, info, verbose=verbose)

    clusterer = AgglomerativeClustering(
        metric="precomputed",
        linkage="average",
        distance_threshold=threshold,
        n_clusters=None,
    )
    labels = clusterer.fit_predict(1 - sim_matrix)
    info = info.copy()
    info["Cluster"] = labels

    topics_to_merge = (
        info.groupby("Cluster")["Topic"]
        .apply(list)
        .loc[lambda x: x.apply(len) > 1]
        .tolist()
    )

    if verbose:
        print(f"\nMerging {len(topics_to_merge)} groups:")
        info["claude_label"] = info["Claude"].apply(
            lambda x: x[0][0] if isinstance(x[0], tuple) else str(x[0])
        )
        for group in topics_to_merge:
            group_labels = info[info["Topic"].isin(group)]["claude_label"].tolist()
            print(f"  {group} → {' | '.join(group_labels)}")

    if topics_to_merge:
        topic_model.merge_topics(review_texts, topics_to_merge)

    return topic_model


# ---------------------------------------------------------------------------
# Representation labelling
# ---------------------------------------------------------------------------

def apply_representation_models(
    topic_model: BERTopic,
    review_texts: list[str],
    vectorizer: CountVectorizer,
    client,
) -> BERTopic:
    """Apply KeyBERT + Claude representation models to label topics."""
    representation_model = {
        "KeyBERT": KeyBERTInspired(),
        "Claude": ClaudeRepresentation(client),
    }
    topic_model.update_topics(
        review_texts,
        vectorizer_model=vectorizer,
        representation_model=representation_model,
    )
    return topic_model


# ---------------------------------------------------------------------------
# Full modelling pipeline
# ---------------------------------------------------------------------------

def run_topic_model(
    review_texts: list[str],
    embeddings: np.ndarray,
    vectorizer: CountVectorizer,
    embedding_model,
    client,
    verbose: bool = True,
) -> tuple[BERTopic, list]:
    """
    Full topic modelling pipeline:
      fit → reduce outliers → label → merge → relabel
    Returns (topic_model, final_topics).
    """
    # 1. Fit
    topic_model, topics, probs = fit_until_stable(
        review_texts, embeddings, vectorizer, embedding_model, verbose=verbose
    )

    # 2. Reduce outliers
    new_topics = reduce_outliers(topic_model, review_texts, topics, vectorizer)

    # 3. Initial representation (needed for similarity matrix)
    if verbose:
        print("\nApplying initial representation models...")
    topic_model = apply_representation_models(
        topic_model, review_texts, vectorizer, client
    )

    # 4. Merge similar topics
    if verbose:
        print("\nMerging similar topics...")
    info = topic_model.get_topic_info()
    info = info[info["Topic"] != -1].reset_index(drop=True)
    topic_model = merge_similar_topics(
        topic_model, review_texts, info, embedding_model, verbose=verbose
    )

    # 5. Final representation on merged topics
    if verbose:
        print("\nApplying final representation models...")
    topic_model = apply_representation_models(
        topic_model, review_texts, vectorizer, client
    )

    final_topics = topic_model.topics_
    return topic_model, final_topics
