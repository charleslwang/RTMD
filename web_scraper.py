# utils/web_scraper.py
import requests
from bs4 import BeautifulSoup
from transformers import pipeline

# Initialize the summarization and sentiment pipelines from transformers
summarizer = pipeline("summarization")
sentiment_analyzer = pipeline("sentiment-analysis")

def get_news(company_name):
    api_key = "API_KEY"  # Replace with your News API key
    url = f"https://newsapi.org/v2/everything?q={company_name}&sortBy=publishedAt&language=en&apiKey={api_key}"
    response = requests.get(url)
    data = response.json()

    # Check if the request was successful
    if response.status_code == 200 and "articles" in data:
        articles = []
        for article in data["articles"][:5]:  # Limit to 5 articles
            articles.append({
                "title": article["title"],
                "link": article["url"],
                "source": article["source"]["name"]
            })
        return articles
    else:
        print("Error fetching news:", data.get("message", "Unknown error"))
        return []


def summarize_news(articles):
    summaries = []
    for article in articles:
        summary = summarizer(article["title"], max_length=60, min_length=20, do_sample=False)[0]['summary_text']
        summaries.append({"title": article["title"], "link": article["link"], "summary": summary})
    return summaries


def analyze_sentiment(articles):
    sentiments = []
    for article in articles:
        sentiment = sentiment_analyzer(article["title"])[0]
        sentiments.append({"title": article["title"], "link": article["link"], "sentiment": sentiment["label"],
                           "score": sentiment["score"]})
    return sentiments
