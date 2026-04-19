import sys

# Simple rule-based fallback lexicon
POSITIVE_WORDS = {'love', 'fantastic', 'great', 'good', 'excellent', 'amazing', 'happy', 'wonderful', 'best', 'fascinating', 'useful'}
NEGATIVE_WORDS = {'worst', 'bad', 'terrible', 'awful', 'hate', 'poor', 'disappointed', 'boring', 'useless', 'difficult'}

def setup_nltk():
    """Attempt to download required NLTK data if nltk is available."""
    try:
        import nltk
        try:
            nltk.data.find('sentiment/vader_lexicon.zip')
        except LookupError:
            print("[Info] Downloading VADER lexicon...")
            nltk.download('vader_lexicon', quiet=True)
        return True
    except ImportError:
        print("[Warning] NLTK not found. Using simple rule-based analyzer.")
        return False

def analyze_sentiment_nltk(text):
    """Analyze text using NLTK VADER."""
    from nltk.sentiment import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = sia.polarity_scores(text)
    if sentiment_scores['compound'] >= 0.05:
        return "Positive"
    elif sentiment_scores['compound'] <= -0.05:
        return "Negative"
    else:
        return "Neutral"

def analyze_sentiment_simple(text):
    """Simple rule-based sentiment analysis fallback."""
    words = text.lower().split()
    score = 0
    for word in words:
        clean_word = "".join(filter(str.isalpha, word))
        if clean_word in POSITIVE_WORDS:
            score += 1
        elif clean_word in NEGATIVE_WORDS:
            score -= 1
    
    if score > 0:
        return "Positive"
    elif score < 0:
        return "Negative"
    else:
        return "Neutral"

def main():
    has_nltk = setup_nltk()
    
    sample_reviews = [
        "I absolutely love this product! It's fantastic.",
        "This is the worst experience I have ever had.",
        "The product is okay, it does the job but nothing special.",
        "Machine learning is fascinating and highly useful!",
        "I am very disappointed with the poor quality."
    ]
    
    print("\n" + "="*40)
    print("      SENTIMENT ANALYSIS RESULTS")
    print("="*40)
    
    for review in sample_reviews:
        if has_nltk:
            result = analyze_sentiment_nltk(review)
        else:
            result = analyze_sentiment_simple(review)
        
        print(f"Text:      \"{review}\"")
        print(f"Sentiment: {result}")
        print("-" * 40)

if __name__ == "__main__":
    main()
