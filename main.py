# main.py
from sentiment_analyzer import SentimentAnalyzer
from privacy_utils import PrivacyPreserver
from app import create_app

def main():
    # Initialize components
    analyzer = SentimentAnalyzer()
    privacy_preserver = PrivacyPreserver()
    
    # Create and run the Flask app
    app = create_app(analyzer, privacy_preserver)
    app.run(debug=True)

if __name__ == "__main__":
    main()
