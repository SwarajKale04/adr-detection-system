"""
Train MLP Sentiment Model Script.

Trains the MLP sentiment analyzer on WebMD review data and saves the model.
Provides comparison with rule-based approach.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from backend.modules.mlp_sentiment_analyzer import MLPSentimentAnalyzer, train_mlp_model, compare_with_rule_based
from backend.modules.sentiment_analyzer import SentimentAnalyzer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np


def plot_training_comparison():
    """Plot comparison between MLP and rule-based approaches."""
    # Load sample data for comparison
    df = pd.read_excel("backend/data/DS1_WebMD_Reviews.xlsx")
    sample_texts = df['text'].dropna().head(200).tolist()

    # Initialize analyzers
    mlp_analyzer = MLPSentimentAnalyzer()
    if not mlp_analyzer.load_model():
        print("MLP model not found. Please train it first.")
        return

    rule_analyzer = SentimentAnalyzer()

    # Get predictions
    mlp_predictions = []
    rule_predictions = []

    print("Comparing predictions on 200 samples...")

    for text in sample_texts:
        mlp_result = mlp_analyzer.predict_sentiment(text)
        rule_result = rule_analyzer.analyze_sentiment(text)

        mlp_predictions.append(mlp_result['sentiment'])
        rule_predictions.append(rule_result['sentiment'])

    # Create confusion matrix style comparison
    sentiments = ['positive', 'negative']
    cm_data = np.zeros((2, 2))

    for mlp, rule in zip(mlp_predictions, rule_predictions):
        mlp_idx = 0 if mlp == 'positive' else 1
        rule_idx = 0 if rule == 'positive' else 1
        cm_data[mlp_idx, rule_idx] += 1

    # Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_data, annot=True, fmt='.0f', cmap='Blues',
                xticklabels=['Rule: Positive', 'Rule: Negative'],
                yticklabels=['MLP: Positive', 'MLP: Negative'])
    plt.title('MLP vs Rule-Based Sentiment Analysis\n(Agreement Matrix)')
    plt.tight_layout()
    plt.savefig('backend/output/sentiment_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Calculate agreement statistics
    agreements = sum(1 for mlp, rule in zip(mlp_predictions, rule_predictions) if mlp == rule)
    agreement_rate = agreements / len(mlp_predictions)

    print("\nComparison Results:")
    print(f"Total samples: {len(mlp_predictions)}")
    print(f"Agreements: {agreements}")
    print(f"Agreement rate: {agreement_rate:.3f}")
    print(f"MLP Positive: {mlp_predictions.count('positive')}")
    print(f"MLP Negative: {mlp_predictions.count('negative')}")
    print(f"Rule Positive: {rule_predictions.count('positive')}")
    print(f"Rule Negative: {rule_predictions.count('negative')}")


def main():
    """Main training function."""
    print("🔥 MLP Sentiment Analyzer Training")
    print("=" * 50)

    # Check if data exists
    data_path = "backend/data/DS1_WebMD_Reviews.xlsx"
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        return

    # Train the model
    print("🚀 Starting MLP training...")
    results = train_mlp_model(data_path)

    print("\n📊 Training Results:")
    print(f"✅ Accuracy: {results['accuracy']:.3f}")
    print(f"📝 Training samples: {results['training_samples']}")
    print(f"🧪 Test samples: {results['test_samples']}")
    print(f"🔢 Features: {results['feature_count']}")

    print("\n📋 Classification Report:")
    print(results['classification_report'])

    # Save results
    with open("backend/output/mlp_training_results.txt", "w") as f:
        f.write("MLP Sentiment Analyzer Training Results\n")
        f.write("=" * 50 + "\n")
        f.write(f"Accuracy: {results['accuracy']:.3f}\n")
        f.write(f"Training samples: {results['training_samples']}\n")
        f.write(f"Test samples: {results['test_samples']}\n")
        f.write(f"Features: {results['feature_count']}\n\n")
        f.write("Classification Report:\n")
        f.write(results['classification_report'])

    print("💾 Results saved to: backend/output/mlp_training_results.txt")
    print("🤖 Model saved to: backend/models/mlp_sentiment_model.pkl")
    print("📊 Vectorizer saved to: backend/models/tfidf_vectorizer.pkl")

    # Optional: Run comparison
    run_comparison = input("\n🔍 Run comparison with rule-based approach? (y/n): ").lower().strip()
    if run_comparison == 'y':
        plot_training_comparison()


if __name__ == "__main__":
    main()