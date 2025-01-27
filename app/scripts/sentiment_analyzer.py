# scripts/sentiment_analyzer.py

import warnings
from transformers import pipeline, TFDistilBertForSequenceClassification, DistilBertTokenizerFast
from newsapi import NewsApiClient
from datetime import datetime, timedelta

class SentimentAnalyzer:
    def __init__(self, news_api_key):
        # Suppress warnings
        warnings.filterwarnings('ignore')
        
        # Specify model name and revision
        model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        revision = "af0f99b"
        
        # Initialize tokenizer and model explicitly
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_name, revision=revision)
        self.model = TFDistilBertForSequenceClassification.from_pretrained(
            model_name,
            revision=revision,
            from_pt=True
        )
        
        # Initialize sentiment pipeline with specific model and tokenizer
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model=self.model,
            tokenizer=self.tokenizer
        )
        
        # Initialize NewsAPI client
        self.news_api = NewsApiClient(api_key=news_api_key)
    
    def get_news(self, company, days=7):
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        from_date = start_date.strftime('%Y-%m-%d')
        to_date = end_date.strftime('%Y-%m-%d')
        
        try:
            news = self.news_api.get_everything(
                q=company,
                language='en',
                sort_by='publishedAt',
                from_param=from_date,
                to=to_date
            )
            return news['articles']
        except Exception as e:
            print(f"Error fetching news: {str(e)}")
            return []
    
    def analyze_sentiment(self, text):
        try:
            result = self.sentiment_analyzer(text)[0]
            return {
                'label': result['label'],
                'score': result['score']
            }
        except Exception as e:
            print(f"Error analyzing sentiment: {str(e)}")
            return {
                'label': 'NEUTRAL',
                'score': 0.5
            }
    
    def get_company_sentiment(self, company, days=7):
        articles = self.get_news(company, days)
        sentiments = []
        
        for article in articles:
            if article['title']:  # Check if title exists
                sentiment = self.analyze_sentiment(article['title'])
                sentiments.append({
                    'title': article['title'],
                    'sentiment': sentiment['label'],
                    'score': sentiment['score'],
                    'url': article.get('url', ''),
                    'published_at': article.get('publishedAt', '')
                })
        
        return sentiments