"""
Test BioBERT Sentiment Analyzer.

Quick test script to verify BioBERT integration works correctly.
"""

import sys
import os
sys.path.append('backend')

from modules.biobert_sentiment_analyzer import BioBERTSentimentAnalyzer


def test_biobert():
    """Test BioBERT sentiment analysis on sample medical reviews."""
    print("🧠 Testing BioBERT Sentiment Analyzer")
    print("=" * 50)

    # Initialize analyzer
    analyzer = BioBERTSentimentAnalyzer()

    # Test loading
    if not analyzer.load_model():
        print("❌ Failed to load BioBERT model")
        return False

    print("✅ BioBERT model loaded successfully")

    # Test reviews
    test_reviews = [
        "This medication completely cured my migraine headaches. Highly effective!",
        "Experienced severe nausea and vomiting after taking this drug. Never again.",
        "Mild improvement in my arthritis pain, but the side effects are manageable.",
        "This antibiotic cleared my infection quickly with no adverse reactions.",
        "The drug caused terrible dizziness and I had to stop taking it immediately."
    ]

    print(f"\n🧪 Testing on {len(test_reviews)} sample reviews:")
    print("-" * 50)

    results = []
    for i, review in enumerate(test_reviews, 1):
        print(f"\nReview {i}: {review[:50]}...")

        result = analyzer.analyze_sentiment(review)
        results.append(result)

        print(f"  Sentiment: {result['sentiment'].upper()}")
        print(f"  Confidence: {result['confidence']:.3f}")
        print(f"  Scores: {result['scores']}")

    # Summary
    positive_count = sum(1 for r in results if r['sentiment'] == 'positive')
    negative_count = sum(1 for r in results if r['sentiment'] == 'negative')

    print(f"\n📊 Summary:")
    print(f"   Positive: {positive_count}")
    print(f"   Negative: {negative_count}")
    print(f"   Average Confidence: {sum(r['confidence'] for r in results) / len(results):.3f}")

    print("\n✅ BioBERT test completed successfully!")
    return True


if __name__ == "__main__":
    test_biobert()