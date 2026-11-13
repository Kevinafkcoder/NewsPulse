import streamlit as st
import plotly.express as px
import pandas as pd

from database import (
    create_table, get_articles, get_article_count, get_topic_history
)
from news_collector import collect_news
from preprocessing import prepare_articles
from topic_detection import detect_topics
from trend_analysis import analyze_trends
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity 

st.set_page_config(page_title="NewsPulse", page_icon="📰", layout="wide")
create_table()

@st.cache_data(ttl=300)
def load_analysis():
    raw_df = get_articles()

    if raw_df.empty:
        return raw_df, pd.DataFrame(), [], pd.DataFrame(), pd.DataFrame()

    processed = prepare_articles(raw_df)

    if len(processed) < 3:
        return raw_df, pd.DataFrame(), [], pd.DataFrame(), pd.DataFrame()

    clustered, topic_names = detect_topics(processed, n_clusters=None)
    summary, daily = analyze_trends(clustered)

    return raw_df, clustered, topic_names, summary, daily

st.title("📰 NewsPulse")
st.subheader("Online News Topic Detection and Trend Analysis")
st.caption("Discover news topics, identify emerging coverage and explore how attention changes over time.")

st.sidebar.header("Controls")
if st.sidebar.button("Fetch Latest News", width="stretch"):
    with st.spinner("Collecting news articles..."):
        try:
            collect_news()
            st.cache_data.clear()
            st.sidebar.success("News collection finished. Refreshing analysis...")
            st.rerun()
        except Exception as exc:
            st.sidebar.error(f"Collection failed: {exc}")

if st.sidebar.button("Refresh Analysis", width="stretch"):
    st.cache_data.clear()
    st.rerun()

try:
    with st.spinner("Loading NewsPulse analysis..."):
        raw_df, clustered, topic_names, summary, daily = load_analysis()
except Exception as exc:
    st.error(f"Analysis failed: {exc}")
    st.stop()

if raw_df.empty:
    st.info("No articles found. Use Fetch Latest News to collect articles.")
    st.stop()

if clustered.empty:
    st.warning("At least three valid articles are required for topic detection.")
    st.stop()

# Sidebar filters

st.sidebar.subheader("Article filters")
search_text = st.sidebar.text_input("Search headlines and descriptions")
sources = sorted(clustered["source"].dropna().astype(str).unique()) if "source" in clustered else []
categories = sorted(clustered["category"].dropna().astype(str).unique()) if "category" in clustered else []
selected_sources = st.sidebar.multiselect("Source", sources, default=[])
selected_categories = st.sidebar.multiselect("Category", categories, default=[])

date_series = pd.to_datetime(clustered["published_at"], errors="coerce", utc=True)
min_date = date_series.min().date() if date_series.notna().any() else None
max_date = date_series.max().date() if date_series.notna().any() else None
date_range = None
if min_date and max_date:
    date_range = st.sidebar.date_input(
        "Publication date range", value=(min_date, max_date),
        min_value=min_date, max_value=max_date
    )

filtered = clustered.copy()
if search_text.strip():
    mask = (
        filtered["title"].fillna("").str.contains(search_text, case=False, regex=False) |
        filtered["description"].fillna("").str.contains(search_text, case=False, regex=False)
    )
    filtered = filtered[mask]
if selected_sources:
    filtered = filtered[filtered["source"].astype(str).isin(selected_sources)]
if selected_categories:
    filtered = filtered[filtered["category"].astype(str).isin(selected_categories)]
if isinstance(date_range, tuple) and len(date_range) == 2:
    dates = pd.to_datetime(filtered["published_at"], errors="coerce", utc=True).dt.date
    filtered = filtered[(dates >= date_range[0]) & (dates <= date_range[1])]

# Overview metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total articles", get_article_count())
c2.metric("Articles analysed", len(clustered))
c3.metric("Topics discovered", clustered["topic_key"].nunique())
c4.metric("Articles after filters", len(filtered))

st.divider()
st.header("🔥 Topic trends")
if not summary.empty:
    cols = ["topic_name", "recent_articles", "previous_articles",
            "recent_share", "previous_share", "share_change", "trend"]
    st.dataframe(
        summary[cols].rename(columns={
            "topic_name": "Topic", "recent_articles": "Recent articles",
            "previous_articles": "Previous articles", "recent_share": "Recent share (%)",
            "previous_share": "Previous share (%)", "share_change": "Share change (pp)",
            "trend": "Status"
        }).round(2),
        width="stretch", hide_index=True
    )
else:
    st.info("Not enough publication-date data to calculate trends.")

st.header("📊 Topic distribution")
distribution = clustered.groupby("topic_name").size().reset_index(name="Articles")
fig = px.bar(distribution, x="topic_name", y="Articles", color="topic_name",
             title="Articles per persistent topic")
st.plotly_chart(fig, width="stretch")

st.header("📈 Topic popularity over time")
if not daily.empty:
    fig2 = px.line(daily, x="date", y="article_count", color="topic_name",
                   markers=True, title="Daily news coverage by topic")
    st.plotly_chart(fig2, width="stretch")
else:
    st.info("No dated topic history is available.")

st.header("🔎 Search and explore articles")
topic_options = ["All topics"] + sorted(filtered["topic_name"].dropna().unique().tolist())
selected_topic = st.selectbox("Filter by discovered topic", topic_options)

if selected_topic != "All topics":
    #st.caption(f"Full topic name: {selected_topic}")
    filtered = filtered[filtered["topic_name"] == selected_topic]

st.caption(f"Showing {len(filtered)} matching articles after applying all filters.")


if not filtered.empty:

    # Pagination: display 10 articles per page
    sorted_articles = filtered.sort_values(
        "published_at", ascending=False
    )

    page_size = 10
    total_articles = len(sorted_articles)
    total_pages = (total_articles + page_size - 1) // page_size

    page = st.number_input(
        "Page",
        min_value=1,
        max_value=total_pages,
        value=1,
        step=1,
        key="article_page"
    )

    start = (page - 1) * page_size
    end = start + page_size

    page_articles = sorted_articles.iloc[start:end]

    st.caption(
        f"Showing {start + 1}–{min(end, total_articles)} "
        f"of {total_articles} articles"
    )

    for _, article in page_articles.iterrows():
        title = str(article.get("title", "Untitled"))
        with st.expander(title):
            description = article.get("description")
            if pd.notna(description) and str(description).strip():
                st.write(description)
            st.caption(
                f"Topic: {article.get('topic_name', 'N/A')} | "
                f"Source: {article.get('source', 'N/A')} | "
                f"Category: {article.get('category', 'N/A')} | "
                f"Published: {article.get('published_at', 'N/A')}"
            )
            if article.get("is_representative", False):
                st.caption("⭐ Representative article for this topic")
            url = article.get("url")
            if isinstance(url, str) and url.startswith(("http://", "https://")):
                st.markdown(f"[Read original article]({url})")
else:
    st.info("No articles match the selected search and filters.")

st.header("🧭 Explore Detected Topics")
topic_detail = st.selectbox(
    "Choose a topic to inspect",
    sorted(clustered["topic_name"].dropna().unique().tolist()),
    key="topic_detail"
)
topic_rows = clustered[clustered["topic_name"] == topic_detail].copy()
st.write(f"**{len(topic_rows)} articles** in this topic.")
if "topic_keywords" in topic_rows and not topic_rows.empty:
    st.caption("Keywords: " + str(topic_rows["topic_keywords"].iloc[0]))

representatives = topic_rows.sort_values("representative_rank").head(3)
st.markdown("**Representative articles**")
for _, row in representatives.iterrows():
    st.write(f"• {row['title']}")

# DOWNLOAD ARTICLES AS CSV

st.divider()
st.subheader("📥 Download Articles")

if not filtered.empty:

    export_columns = [
        col for col in [
            "title",
            "description",
            "source",
            "category",
            "published_at",
            "topic_name",
            "url"
        ]
        if col in filtered.columns
    ]

    csv_data = filtered[export_columns].to_csv(
        index=False
    ).encode("utf-8-sig")

    st.download_button(
        label="Download Articles (CSV)",
        data=csv_data,
        file_name="newspulse_articles.csv",
        mime="text/csv"
    )

else:
    st.info("No articles available to download.")


st.divider()
st.caption("NewsPulse | Python · GNews · PostgreSQL · Sentence Transformers · K-Means · Streamlit")