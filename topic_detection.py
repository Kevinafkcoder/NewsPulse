import uuid
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from database import get_topic_registry, save_topic_assignments

_model = None


def get_model():
    global _model
    if _model is None:
        print("Loading Sentence Transformer model...")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        print("Model loaded successfully.")
    return _model


def remove_near_duplicates(df, model, threshold=0.90):
    """Remove near-duplicate titles from this analysis only; database rows remain unchanged."""
    df = df.copy().reset_index(drop=True)
    if len(df) < 2:
        return df

    titles = df["title"].fillna("").astype(str).tolist()
    title_embeddings = model.encode(
        titles, convert_to_numpy=True, normalize_embeddings=True,
        show_progress_bar=False
    )
    similarity_matrix = title_embeddings @ title_embeddings.T
    removed = set()

    for i in range(len(df)):
        if i in removed:
            continue
        for j in range(i + 1, len(df)):
            if j not in removed and float(similarity_matrix[i, j]) >= threshold:
                removed.add(j)

    return df.drop(index=list(removed)).reset_index(drop=True)


def _cluster_keywords(texts, max_words=5):
    try:
        vectorizer = TfidfVectorizer(
            stop_words="english", ngram_range=(1, 2),
            max_features=3000, max_df=0.95
        )
        matrix = vectorizer.fit_transform(texts)
        scores = np.asarray(matrix.mean(axis=0)).ravel()
        terms = vectorizer.get_feature_names_out()
        order = scores.argsort()[::-1]
        return [terms[i] for i in order[:max_words] if scores[i] > 0]
    except ValueError:
        return []


def _make_label(keywords, representative_title, cluster_number):
    if keywords:
        return " · ".join(keywords[:3]).title()

    return f"News Topic {cluster_number + 1}"


def _match_persistent_topics(centroids, labels, df, embeddings, threshold=0.62):
    """
    Greedily match current cluster centroids to the previous registry.
    A new stable topic key is created when no sufficiently similar old topic exists.
    """
    registry = get_topic_registry()
    old_vectors = []
    for item in registry:
        vector = np.asarray(item["centroid"], dtype=float)
        if vector.shape == centroids[0].shape:
            old_vectors.append((item, vector))

    candidates = []
    for cluster_id, centroid in enumerate(centroids):
        for item, old_vector in old_vectors:
            similarity = float(cosine_similarity(
                centroid.reshape(1, -1), old_vector.reshape(1, -1)
            )[0, 0])
            candidates.append((similarity, cluster_id, item))

    candidates.sort(reverse=True, key=lambda x: x[0])
    matched_clusters = set()
    matched_keys = set()
    cluster_to_key = {}

    for similarity, cluster_id, item in candidates:
        if similarity < threshold:
            break
        if cluster_id in matched_clusters or item["topic_key"] in matched_keys:
            continue
        cluster_to_key[cluster_id] = item["topic_key"]
        matched_clusters.add(cluster_id)
        matched_keys.add(item["topic_key"])

    for cluster_id in range(len(centroids)):
        if cluster_id not in cluster_to_key:
            cluster_to_key[cluster_id] = str(uuid.uuid4())

    return cluster_to_key


def detect_topics(df, n_clusters=None, min_clusters=3, max_clusters=8):
    df = df.copy()

    required = {"clean_text", "title"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    df["clean_text"] = df["clean_text"].fillna("").astype(str)
    df = df[df["clean_text"].str.strip() != ""].reset_index(drop=True)

    if len(df) < 3:
        raise ValueError("At least 3 valid articles are required for topic detection.")

    model = get_model()
    df = remove_near_duplicates(df, model, threshold=0.90)

    if len(df) < 3:
        raise ValueError("At least 3 articles are required after near-duplicate removal.")

    embeddings = model.encode(
        df["clean_text"].tolist(), show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True
    )

    # Select K using cosine silhouette score.
    if n_clusters is None:
        best_score, best_k = -1, None
        upper_k = min(max_clusters, len(df) - 1)
        for k in range(min_clusters, upper_k + 1):
            model_k = KMeans(n_clusters=k, random_state=42, n_init=20, max_iter=300)
            labels_k = model_k.fit_predict(embeddings)
            if len(set(labels_k)) < 2 or len(set(labels_k)) >= len(df):
                continue
            score = silhouette_score(embeddings, labels_k, metric="cosine")
            print(f"K={k}, Silhouette Score={score:.4f}")
            if score > best_score:
                best_score, best_k = score, k
        if best_k is None:
            raise ValueError("Unable to determine a suitable number of clusters.")
        n_clusters = best_k
        print(f"Selected K={n_clusters}; silhouette={best_score:.4f}")

    final_model = KMeans(
        n_clusters=int(n_clusters), random_state=42,
        n_init=20, max_iter=300
    )
    df["cluster_id"] = final_model.fit_predict(embeddings)
    centroids = final_model.cluster_centers_

    cluster_to_key = _match_persistent_topics(
        centroids, df["cluster_id"].to_numpy(), df, embeddings
    )

    topic_names = {}
    topic_records = []
    assignments = []

    for cluster_id in sorted(df["cluster_id"].unique()):
        indices = np.where(df["cluster_id"].to_numpy() == cluster_id)[0]
        cluster_embeddings = embeddings[indices]
        centroid = centroids[cluster_id]

        # Representative articles are nearest to the cluster centroid.
        similarities = cluster_embeddings @ centroid / (
            np.linalg.norm(centroid) * np.linalg.norm(cluster_embeddings, axis=1) + 1e-12
        )
        ranked_local = np.argsort(similarities)[::-1]
        representative_index = indices[ranked_local[0]]
        representative_title = df.iloc[representative_index]["title"]

        keywords = _cluster_keywords(
            df.iloc[indices]["clean_text"].tolist(), max_words=5
        )
        label = _make_label(keywords, representative_title, int(cluster_id))
        topic_key = cluster_to_key[int(cluster_id)]

        topic_names[int(cluster_id)] = label
        df.loc[df["cluster_id"] == cluster_id, "topic_key"] = topic_key
        df.loc[df["cluster_id"] == cluster_id, "topic_name"] = label
        df.loc[df["cluster_id"] == cluster_id, "topic_keywords"] = ", ".join(keywords)
        df.loc[df["cluster_id"] == cluster_id, "is_representative"] = False
        df.loc[representative_index, "is_representative"] = True

        topic_records.append({
            "topic_key": topic_key,
            "topic_name": label,
            "centroid": centroid.astype(float).tolist(),
        })

        if "id" in df.columns:
            for article_id in df.iloc[indices]["id"].dropna().tolist():
                assignments.append((int(article_id), topic_key))

    if assignments:
        save_topic_assignments(topic_records, assignments)

    # Representative article ranking within each topic.
    df["representative_rank"] = 0
    for cluster_id in sorted(df["cluster_id"].unique()):
        indices = np.where(df["cluster_id"].to_numpy() == cluster_id)[0]
        centroid = centroids[cluster_id]
        sims = embeddings[indices] @ centroid / (
            np.linalg.norm(centroid) * np.linalg.norm(embeddings[indices], axis=1) + 1e-12
        )
        order = np.argsort(sims)[::-1]
        for rank, local_index in enumerate(order, start=1):
            df.loc[indices[local_index], "representative_rank"] = rank

    df["is_representative"] = df["representative_rank"] <= 3
    return df, topic_names
