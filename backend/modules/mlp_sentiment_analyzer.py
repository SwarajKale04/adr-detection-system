"""
MLP Sentiment Analyzer Module.

Provides a machine learning approach to sentiment analysis using Multi-Layer Perceptron.
Trains on WebMD review data and provides sentiment classification as an alternative
to the rule-based sentiment analyzer.

This demonstrates NLP capabilities with traditional machine learning techniques.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle
import os
from typing import Tuple, Optional


class MLPSentimentAnalyzer:
    """
    Multi-Layer Perceptron based sentiment analyzer.

    Trains an MLP classifier on TF-IDF features from text reviews
    to predict positive/negative sentiment.
    """

    def __init__(self, model_path: str = "backend/models/mlp_sentiment_model.pkl",
                 vectorizer_path: str = "backend/models/tfidf_vectorizer.pkl"):
        """
        Initialize the MLP sentiment analyzer.

        Args:
            model_path (str): Path to save/load the trained model
            vectorizer_path (str): Path to save/load the TF-IDF vectorizer
        """
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.model = None
        self.vectorizer = None
        self.is_trained = False

        # Ensure models directory exists
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

    def preprocess_text(self, text: str) -> str:
        """
        Basic text preprocessing for MLP input.

        Args:
            text (str): Raw text to preprocess

        Returns:
            str: Preprocessed text
        """
        if not isinstance(text, str):
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove extra whitespace
        text = ' '.join(text.split())

        return text

    def prepare_training_data(self, reviews_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data from reviews dataframe.

        Args:
            reviews_df (pd.DataFrame): DataFrame with 'text' and sentiment labels

        Returns:
            Tuple[np.ndarray, np.ndarray]: X (features), y (labels)
        """
        # Preprocess texts
        texts = reviews_df['text'].apply(self.preprocess_text).tolist()

        # For demonstration, we'll use a simple rule to create labels
        # In a real scenario, you'd have labeled data
        labels = []
        for text in texts:
            # Simple heuristic: if text contains more positive than negative words
            positive_words = ['good', 'great', 'excellent', 'effective', 'helpful', 'better', 'well', 'fine']
            negative_words = ['bad', 'terrible', 'awful', 'worst', 'horrible', 'pain', 'sick', 'nausea', 'headache']

            pos_count = sum(1 for word in positive_words if word in text)
            neg_count = sum(1 for word in negative_words if word in text)

            # Label as positive if more positive words, negative otherwise
            label = 1 if pos_count > neg_count else 0
            labels.append(label)

        return np.array(texts), np.array(labels)

    def train(self, reviews_df: pd.DataFrame, test_size: float = 0.2) -> dict:
        """
        Train the MLP sentiment classifier.

        Args:
            reviews_df (pd.DataFrame): DataFrame with review texts
            test_size (float): Proportion of data for testing

        Returns:
            dict: Training results and metrics
        """
        print("Preparing training data...")
        X_texts, y = self.prepare_training_data(reviews_df)

        # Split data
        X_train_texts, X_test_texts, y_train, y_test = train_test_split(
            X_texts, y, test_size=test_size, random_state=42, stratify=y
        )

        print(f"Training data: {len(X_train_texts)} samples")
        print(f"Test data: {len(X_test_texts)} samples")
        print(f"Positive samples: {sum(y_train)}/{len(y_train)} ({sum(y_train)/len(y_train)*100:.1f}%)")

        # Create TF-IDF vectorizer
        print("Creating TF-IDF features...")
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=2
        )

        X_train = self.vectorizer.fit_transform(X_train_texts)
        X_test = self.vectorizer.transform(X_test_texts)

        print(f"Feature matrix shape: {X_train.shape}")

        # Create and train MLP classifier
        print("Training MLP classifier...")
        self.model = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            alpha=0.001,
            batch_size=32,
            learning_rate='adaptive',
            max_iter=200,
            random_state=42,
            verbose=True
        )

        self.model.fit(X_train, y_train)

        # Evaluate
        print("Evaluating model...")
        y_pred = self.model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=['Negative', 'Positive'])

        print(f"\nAccuracy: {accuracy:.3f}")
        print("\nClassification Report:")
        print(report)

        # Save model and vectorizer
        print("Saving model and vectorizer...")
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)

        with open(self.vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)

        self.is_trained = True

        return {
            'accuracy': accuracy,
            'classification_report': report,
            'training_samples': len(X_train_texts),
            'test_samples': len(X_test_texts),
            'feature_count': X_train.shape[1]
        }

    def load_model(self) -> bool:
        """
        Load pre-trained model and vectorizer.

        Returns:
            bool: True if successfully loaded
        """
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)

            with open(self.vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)

            self.is_trained = True
            print("MLP model loaded successfully")
            return True

        except FileNotFoundError:
            print("Model files not found. Please train the model first.")
            return False

    def predict_sentiment(self, text: str) -> dict:
        """
        Predict sentiment for a given text.

        Args:
            text (str): Text to analyze

        Returns:
            dict: Sentiment prediction with confidence
        """
        if not self.is_trained:
            if not self.load_model():
                return {'sentiment': 'unknown', 'confidence': 0.0}

        # Preprocess text
        processed_text = self.preprocess_text(text)

        # Vectorize
        text_vector = self.vectorizer.transform([processed_text])

        # Predict
        prediction = self.model.predict(text_vector)[0]
        probabilities = self.model.predict_proba(text_vector)[0]

        sentiment = 'positive' if prediction == 1 else 'negative'
        confidence = float(max(probabilities))

        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'probabilities': {
                'negative': float(probabilities[0]),
                'positive': float(probabilities[1])
            }
        }


def train_mlp_model(data_path: str = "backend/data/DS1_WebMD_Reviews.xlsx") -> dict:
    """
    Convenience function to train MLP model on WebMD data.

    Args:
        data_path (str): Path to WebMD reviews Excel file

    Returns:
        dict: Training results
    """
    # Load data
    df = pd.read_excel(data_path)
    df = df[['drug_name', 'text']].dropna()

    # Initialize and train model
    analyzer = MLPSentimentAnalyzer()
    results = analyzer.train(df)

    return results


def compare_with_rule_based(mlp_analyzer: MLPSentimentAnalyzer,
                          rule_based_analyzer,
                          test_texts: list) -> dict:
    """
    Compare MLP predictions with rule-based predictions.

    Args:
        mlp_analyzer (MLPSentimentAnalyzer): Trained MLP analyzer
        rule_based_analyzer: Rule-based sentiment analyzer module
        test_texts (list): List of texts to compare

    Returns:
        dict: Comparison results
    """
    mlp_predictions = []
    rule_predictions = []

    for text in test_texts[:100]:  # Test on first 100 samples
        mlp_result = mlp_analyzer.predict_sentiment(text)
        rule_result = rule_based_analyzer.analyze_sentiment(text)

        mlp_predictions.append(mlp_result['sentiment'])
        rule_predictions.append(rule_result['sentiment'])

    # Calculate agreement
    agreements = sum(1 for mlp, rule in zip(mlp_predictions, rule_predictions)
                    if mlp == rule)
    agreement_rate = agreements / len(mlp_predictions)

    return {
        'total_samples': len(mlp_predictions),
        'agreements': agreements,
        'agreement_rate': agreement_rate,
        'sample_predictions': list(zip(test_texts[:5], mlp_predictions[:5], rule_predictions[:5]))
    }


if __name__ == "__main__":
    # Example usage
    print("Training MLP Sentiment Analyzer...")
    results = train_mlp_model()
    print(f"Training completed with accuracy: {results['accuracy']:.3f}")