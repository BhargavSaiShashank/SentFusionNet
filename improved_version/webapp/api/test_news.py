import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def test_news():
    analyzer = SentimentIntensityAnalyzer()
    print("Fetching US News...")
    us_ticker = yf.Ticker("^GSPC")
    news = us_ticker.news[:5]
    
    scores = []
    for item in news:
        title = item.get('title', '')
        score = analyzer.polarity_scores(title)['compound']
        print(f"[{score}] {title}")
        scores.append(score)
    
    avg = sum(scores)/len(scores) if scores else 0
    print(f"\nFinal Sentiment Score: {avg}")

if __name__ == "__main__":
    test_news()
