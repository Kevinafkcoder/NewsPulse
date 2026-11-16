# 📰 NewsPulse
### Online News Topic Detection and Trend Analysis

NewsPulse is a Web Content Mining application that uses Natural Language Processing and machine learning to automatically collect online news articles, discover meaningful topics and analyse how news coverage changes over time.

## Objectives

- Collect news articles from different categories.
- Find similar news articles.
- Group articles into topics.
- Extract key keywords and representative articles.
- Track popular and emerging news topics.
- Provide options to search, filter and explore articles.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---| 
| Python | Core development |
| Streamlit | Interactive dashboard |
| PostgreSQL | Article and topic storage |
| GNews API | News collection |
| Sentence Transformers | Semantic embeddings |
| K-Means | Topic clustering |
| Scikit-learn | Similarity and evaluation |
| Pandas | Data processing |
| Plotly | Interactive visualisations |

## ⚙️ Methodology

1. **News Collection:** Fetch articles from different news categories using the GNews API.
2. **Preprocessing:** Clean and prepare article text and remove duplicate and near-duplicate articles.
3. **Semantic Representation:** Generate sentence embeddings using the `all-MiniLM-L6-v2` model.
4. **Topic Detection:** Apply K-Means clustering to group semantically similar articles.
5. **Topic Labelling:** Generate descriptive topic labels and extract representative articles and keywords.
6. **Trend Analysis:** Compare recent and previous article coverage to identify emerging, rising, declining and stable topics.
7. **Visualisation:** Present results through an interactive Streamlit dashboard.

## ✨ Key Features

- Automated news collection
- Exact and near-duplicate detection
- Semantic topic discovery
- Topic distribution visualisation
- Topic popularity over time
- Emerging topic and trend analysis
- Article search and filtering
- Paginated article browsing
- CSV export of articles
- Representative article identification

## 🚀 Installation and Setup

**1. Clone the repository**

```bash
git clone https://github.com/Kevinafkcoder/NewsPulse.git
cd NewsPulse
```

**2. Create and activate a virtual environment**

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Configure environment variables**

Create a `.env` file in the project directory and configure your own credentials:

```env
GNEWS_API_KEY=your_gnews_api_key
DB_HOST=localhost
DB_PORT=5432
DB_NAME=newspulse_db
DB_USER=postgres
DB_PASSWORD=your_database_password
```

**5. Create the PostgreSQL database**

Create a database named `newspulse_db` in PostgreSQL.

**6. Run the application**

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal.

## 📁 Project Structure

```text
NewsPulse/
│
├── app.py
├── database.py
├── news_collector.py
├── preprocessing.py
├── topic_detection.py
├── trend_analysis.py
├── requirements.txt
├── README.md
├── .gitignore
└── data/
```

## 📊 Trend Analysis

NewsPulse compares article coverage across two equal three-day windows:

- **Recent period:** Latest three days.
- **Previous period:** Three days immediately preceding the recent period.

Topics are classified based on changes in article counts and topic share:

- **Emerging:** No articles for the topic in the previous period, but articles in the recent period.
- **Rising:** Topic share increases by at least 5 percentage points.
- **Declining:** Topic share decreases by at least 5 percentage points.
- **Stable:** Topic share changes by less than 5 percentage points.
- **Insufficient Data:** Not enough historical or recent articles for comparison.

## 🎓 Project Information

**Project:** NewsPulse – Online News Topic Detection and Trend Analysis  
**Domain:** Web Content Mining
**Application:** News analytics and topic discovery

