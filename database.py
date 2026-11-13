import os
import json
from contextlib import contextmanager

import pandas as pd
import psycopg2
from dotenv import load_dotenv

load_dotenv()


@contextmanager
def get_connection():
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT", "5432"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
    )
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def create_table():
    """Create the original articles table and additive topic-tracking tables."""
    queries = [
        """
        CREATE TABLE IF NOT EXISTS articles (
            id SERIAL PRIMARY KEY,
            title TEXT NOT NULL,
            description TEXT,
            content TEXT,
            url TEXT UNIQUE NOT NULL,
            published_at TIMESTAMPTZ,
            source TEXT,
            category TEXT,
            collected_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS topic_registry (
            topic_key TEXT PRIMARY KEY,
            topic_name TEXT NOT NULL,
            centroid_json JSONB NOT NULL,
            first_seen TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            last_seen TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS article_topics (
            article_id INTEGER PRIMARY KEY REFERENCES articles(id) ON DELETE CASCADE,
            topic_key TEXT NOT NULL REFERENCES topic_registry(topic_key),
            assigned_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        """,
        "CREATE INDEX IF NOT EXISTS idx_articles_published_at ON articles(published_at);",
        "CREATE INDEX IF NOT EXISTS idx_articles_source ON articles(source);",
        "CREATE INDEX IF NOT EXISTS idx_articles_category ON articles(category);",
        "CREATE INDEX IF NOT EXISTS idx_article_topics_topic_key ON article_topics(topic_key);",
    ]
    with get_connection() as conn:
        with conn.cursor() as cur:
            for query in queries:
                cur.execute(query)


def save_articles(articles):
    query = """
        INSERT INTO articles
        (title, description, content, url, published_at, source, category)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (url) DO NOTHING;
    """
    inserted = 0
    with get_connection() as conn:
        with conn.cursor() as cur:
            for article in articles:
                cur.execute(query, (
                    article.get("title"),
                    article.get("description"),
                    article.get("content"),
                    article.get("url"),
                    article.get("published_at"),
                    article.get("source"),
                    article.get("category"),
                ))
                inserted += cur.rowcount
    return inserted


def get_articles():
    query = """
        SELECT a.id, a.title, a.description, a.content, a.url,
               a.published_at, a.source, a.category, a.collected_at,
               at.topic_key
        FROM articles a
        LEFT JOIN article_topics at ON at.article_id = a.id
        ORDER BY a.published_at DESC NULLS LAST;
    """
    with get_connection() as conn:
        return pd.read_sql_query(query, conn)


def get_article_count():
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM articles;")
            return cur.fetchone()[0]


def get_topic_registry():
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT topic_key, topic_name, centroid_json
                FROM topic_registry
                ORDER BY first_seen;
            """)
            rows = cur.fetchall()
    result = []
    for key, name, centroid in rows:
        if isinstance(centroid, str):
            centroid = json.loads(centroid)
        result.append({
            "topic_key": key,
            "topic_name": name,
            "centroid": centroid,
        })
    return result


def save_topic_assignments(topic_records, assignments):
    """
    topic_records: list of {topic_key, topic_name, centroid}
    assignments: list of (article_id, topic_key)
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            for topic in topic_records:
                cur.execute("""
                    INSERT INTO topic_registry
                        (topic_key, topic_name, centroid_json, first_seen, last_seen)
                    VALUES (%s, %s, %s::jsonb, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                    ON CONFLICT (topic_key) DO UPDATE SET
                        topic_name = EXCLUDED.topic_name,
                        centroid_json = EXCLUDED.centroid_json,
                        last_seen = CURRENT_TIMESTAMP;
                """, (
                    topic["topic_key"],
                    topic["topic_name"],
                    json.dumps(topic["centroid"]),
                ))

            for article_id, topic_key in assignments:
                cur.execute("""
                    INSERT INTO article_topics (article_id, topic_key)
                    VALUES (%s, %s)
                    ON CONFLICT (article_id) DO UPDATE SET
                        topic_key = EXCLUDED.topic_key,
                        assigned_at = CURRENT_TIMESTAMP;
                """, (int(article_id), topic_key))


def get_topic_history():
    query = """
        SELECT tr.topic_key, tr.topic_name, tr.first_seen, tr.last_seen,
               COUNT(at.article_id) AS article_count
        FROM topic_registry tr
        LEFT JOIN article_topics at ON at.topic_key = tr.topic_key
        GROUP BY tr.topic_key, tr.topic_name, tr.first_seen, tr.last_seen
        ORDER BY tr.last_seen DESC;
    """
    with get_connection() as conn:
        return pd.read_sql_query(query, conn)
