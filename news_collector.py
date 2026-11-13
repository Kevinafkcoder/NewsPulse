
import os
import requests
import time
from dotenv import load_dotenv

from database import create_table, save_articles, get_article_count

load_dotenv()

API_KEY = os.getenv("GNEWS_API_KEY")
BASE_URL = "https://gnews.io/api/v4/top-headlines"

CATEGORIES = [
    "general",
    "world",
    "business",
    "technology",
    "sports",
    "science",
    "health"
]


def fetch_news(category):

    if not API_KEY:
        raise ValueError("GNews API key missing in .env")

    params = {
        "apikey": API_KEY,
        "category": category,
        "lang": "en",
        "country": "in",
        "max": 10
    }

    response = requests.get(
        BASE_URL,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    data = response.json()

    articles = []

    for item in data.get("articles", []):

        if not item.get("title") or not item.get("url"):
            continue

        articles.append({
            "title": item.get("title"),
            "description": item.get("description"),
            "content": item.get("content"),
            "url": item.get("url"),
            "published_at": item.get("publishedAt"),
            "source": item.get("source", {}).get("name"),
            "category": category
        })

    return articles


def collect_news():

    create_table()
    all_articles = []

    for category in CATEGORIES:

        try:
            articles = fetch_news(category)
            all_articles.extend(articles)

            print(f"{category}: {len(articles)} fetched")

        except requests.exceptions.RequestException as e:
            print(f"{category} failed: {e}")

        time.sleep(2)

    if all_articles:

        inserted = save_articles(all_articles)

        print("\nCollection completed!")
        print("Fetched:", len(all_articles))
        print("New articles stored:", inserted)
        print("Total articles:", get_article_count())

    else:
        print("No articles collected.")


if __name__ == "__main__":
    collect_news()
