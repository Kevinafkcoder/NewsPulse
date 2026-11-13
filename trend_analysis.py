import pandas as pd


def analyze_trends(df, recent_days=3, minimum_recent_articles=3, share_threshold=5.0):
    """
    Compare equal-length recent and previous windows using persistent topic_key.
    Topic identities remain comparable across dashboard runs once registered.
    """
    if df.empty or "published_at" not in df.columns:
        return pd.DataFrame(), pd.DataFrame()

    df = df.copy()
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    df = df.dropna(subset=["published_at"]).copy()

    identity_col = "topic_key" if "topic_key" in df.columns else "topic_name"
    if identity_col not in df.columns:
        return pd.DataFrame(), pd.DataFrame()

    df[identity_col] = df[identity_col].fillna("")
    df = df[df[identity_col] != ""]
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    latest_date = df["published_at"].max().normalize()
    recent_start = latest_date - pd.Timedelta(days=recent_days - 1)
    previous_start = recent_start - pd.Timedelta(days=recent_days)

    recent = df[
        (df["published_at"] >= recent_start) &
        (df["published_at"] < latest_date + pd.Timedelta(days=1))
    ]
    previous = df[
        (df["published_at"] >= previous_start) &
        (df["published_at"] < recent_start)
    ]

    recent_counts = recent.groupby(identity_col).size()
    previous_counts = previous.groupby(identity_col).size()
    all_topics = recent_counts.index.union(previous_counts.index)

    summary = pd.DataFrame(index=all_topics)
    summary["recent_articles"] = recent_counts.reindex(all_topics, fill_value=0).astype(int)
    summary["previous_articles"] = previous_counts.reindex(all_topics, fill_value=0).astype(int)

    total_recent, total_previous = len(recent), len(previous)
    summary["recent_share"] = (
        summary["recent_articles"] / total_recent * 100 if total_recent else 0.0
    )
    summary["previous_share"] = (
        summary["previous_articles"] / total_previous * 100 if total_previous else 0.0
    )
    summary["share_change"] = summary["recent_share"] - summary["previous_share"]

    name_map = (
        df.drop_duplicates(identity_col).set_index(identity_col)["topic_name"].to_dict()
        if "topic_name" in df.columns else {}
    )
    summary["topic_name"] = [name_map.get(key, str(key)) for key in summary.index]

    def classify(row):
        if total_recent == 0 or total_previous == 0:
            return "Insufficient historical data"
        if row["recent_articles"] < minimum_recent_articles:
            return "Insufficient data"
        if row["previous_articles"] == 0:
            return "Emerging"
        if row["share_change"] >= share_threshold:
            return "Rising"
        if row["share_change"] <= -share_threshold:
            return "Declining"
        return "Stable"

    summary["trend"] = summary.apply(classify, axis=1)
    summary["recent_total"] = total_recent
    summary["previous_total"] = total_previous
    summary = summary.sort_values(
        ["recent_share", "recent_articles"], ascending=False
    ).reset_index(names="topic_key")

    daily = (
        df.groupby([df["published_at"].dt.date, identity_col])
        .size().reset_index(name="article_count")
    )
    daily.columns = ["date", "topic_key", "article_count"]
    daily["topic_name"] = daily["topic_key"].map(name_map).fillna(daily["topic_key"])

    return summary, daily
