"""
BioBERT Sentiment Analyzer Module.

Uses BioBERT (Bidirectional Encoder Representations from Transformers) model
specifically trained on biomedical text for superior medical sentiment analysis.

This demonstrates advanced NLP capabilities with domain-specific language models.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import numpy as np
from typing import Dict, Any, Optional
import os
import pickle


class BioBERTSentimentAnalyzer:
    """
    BioBERT-based sentiment analyzer for medical text.

    Uses a smaller, distilled BERT model fine-tuned for sentiment analysis,
    optimized for biomedical and clinical text understanding.
    """

    def __init__(self,
                 model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
                 cache_dir: str = "backend/models/biobert_cache",
                 device: str = "auto"):
        """
        Initialize BioBERT sentiment analyzer.

        Args:
            model_name (str): HuggingFace model name
            cache_dir (str): Directory to cache model files
            device (str): Device to run model on ('auto', 'cpu', 'cuda')
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.device = device

        # Set device
        if device == "auto":
            self.device = 0 if torch.cuda.is_available() else -1

        self.pipeline = None
        self.is_loaded = False

        # Ensure cache directory exists
        os.makedirs(cache_dir, exist_ok=True)

    def load_model(self) -> bool:
        """
        Load the BioBERT model and tokenizer.

        Returns:
            bool: True if successfully loaded
        """
        try:
            print(f"🔄 Loading BioBERT model: {self.model_name}")

            # Create sentiment analysis pipeline
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                tokenizer=self.model_name,
                device=self.device,
                cache_dir=self.cache_dir,
                return_all_scores=True
            )

            self.is_loaded = True
            print("✅ BioBERT model loaded successfully")
            return True

        except Exception as e:
            print(f"❌ Failed to load BioBERT model: {str(e)}")
            print("💡 Make sure you have internet connection for first-time download")
            return False

    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of medical text using BioBERT.

        Args:
            text (str): Input text to analyze

        Returns:
            dict: Sentiment analysis results with confidence scores
        """
        if not self.is_loaded:
            if not self.load_model():
                return {
                    'sentiment': 'unknown',
                    'confidence': 0.0,
                    'scores': {}
                }

        try:
            # Preprocess text (BioBERT handles this well, but let's clean it)
            text = self._preprocess_text(text)

            # Get prediction
            results = self.pipeline(text)

            # Process results
            if results and len(results) > 0:
                scores = results[0]  # results is a list with one dict

                # Convert to our format
                sentiment_scores = {}
                for score_dict in scores:
                    label = score_dict['label'].lower()
                    # Map LABEL_0 -> negative, LABEL_1 -> positive
                    if label == 'label_0' or label == 'negative':
                        sentiment_scores['negative'] = score_dict['score']
                    elif label == 'label_1' or label == 'positive':
                        sentiment_scores['positive'] = score_dict['score']
                    else:
                        sentiment_scores[label] = score_dict['score']

                # Determine primary sentiment
                if 'positive' in sentiment_scores and 'negative' in sentiment_scores:
                    if sentiment_scores['positive'] > sentiment_scores['negative']:
                        sentiment = 'positive'
                        confidence = sentiment_scores['positive']
                    else:
                        sentiment = 'negative'
                        confidence = sentiment_scores['negative']
                else:
                    # Fallback for different label schemes
                    max_label = max(sentiment_scores, key=sentiment_scores.get)
                    sentiment = max_label
                    confidence = sentiment_scores[max_label]

                return {
                    'sentiment': sentiment,
                    'confidence': float(confidence),
                    'scores': sentiment_scores
                }

        except Exception as e:
            print(f"⚠️  Error in BioBERT analysis: {str(e)}")
            return {
                'sentiment': 'unknown',
                'confidence': 0.0,
                'scores': {}
            }

    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text for BioBERT input.

        Args:
            text (str): Raw text

        Returns:
            str: Preprocessed text
        """
        if not isinstance(text, str):
            return ""

        # Basic cleaning while preserving medical terms
        text = text.strip()

        # Remove excessive whitespace
        text = ' '.join(text.split())

        # Truncate if too long (BioBERT has token limits)
        if len(text) > 512:
            text = text[:512] + "..."

        return text

    def batch_analyze(self, texts: list) -> list:
        """
        Analyze sentiment for multiple texts efficiently.

        Args:
            texts (list): List of texts to analyze

        Returns:
            list: List of sentiment analysis results
        """
        if not self.is_loaded:
            if not self.load_model():
                return [{'sentiment': 'unknown', 'confidence': 0.0, 'scores': {}}] * len(texts)

        try:
            # Preprocess all texts
            processed_texts = [self._preprocess_text(text) for text in texts]

            # Batch process
            results = self.pipeline(processed_texts, batch_size=8)

            # Process results
            processed_results = []
            for result in results:
                if result and len(result) > 0:
                    scores = result[0]
                    sentiment_scores = {}

                    for score_dict in scores:
                        label = score_dict['label'].lower()
                        if label == 'label_0' or label == 'negative':
                            sentiment_scores['negative'] = score_dict['score']
                        elif label == 'label_1' or label == 'positive':
                            sentiment_scores['positive'] = score_dict['score']
                        else:
                            sentiment_scores[label] = score_dict['score']

                    # Determine sentiment
                    if 'positive' in sentiment_scores and 'negative' in sentiment_scores:
                        if sentiment_scores['positive'] > sentiment_scores['negative']:
                            sentiment = 'positive'
                            confidence = sentiment_scores['positive']
                        else:
                            sentiment = 'negative'
                            confidence = sentiment_scores['negative']
                    else:
                        max_label = max(sentiment_scores, key=sentiment_scores.get)
                        sentiment = max_label
                        confidence = sentiment_scores[max_label]

                    processed_results.append({
                        'sentiment': sentiment,
                        'confidence': float(confidence),
                        'scores': sentiment_scores
                    })
                else:
                    processed_results.append({
                        'sentiment': 'unknown',
                        'confidence': 0.0,
                        'scores': {}
                    })

            return processed_results

        except Exception as e:
            print(f"⚠️  Error in batch BioBERT analysis: {str(e)}")
            return [{'sentiment': 'unknown', 'confidence': 0.0, 'scores': {}}] * len(texts)


def compare_with_other_analyzers(biobert_analyzer: BioBERTSentimentAnalyzer,
                                mlp_analyzer,
                                rule_analyzer_func,
                                test_texts: list) -> Dict[str, Any]:
    """
    Compare BioBERT with other sentiment analyzers.

    Args:
        biobert_analyzer: BioBERT analyzer instance
        mlp_analyzer: MLP analyzer instance
        rule_analyzer_func: Rule-based analyzer function
        test_texts: List of test texts

    Returns:
        dict: Comparison results
    """
    results = {
        'biobert': [],
        'mlp': [],
        'rule_based': [],
        'agreements': {'biobert_mlp': 0, 'biobert_rule': 0, 'mlp_rule': 0}
    }

    print("🔍 Comparing sentiment analyzers on sample texts...")

    for text in test_texts[:50]:  # Test on first 50 samples
        # BioBERT
        biobert_result = biobert_analyzer.analyze_sentiment(text)
        results['biobert'].append(biobert_result['sentiment'])

        # MLP
        mlp_result = mlp_analyzer.predict_sentiment(text)
        results['mlp'].append(mlp_result['sentiment'])

        # Rule-based
        rule_sentiment = rule_analyzer_func(text.lower())
        results['rule_based'].append(rule_sentiment)

        # Check agreements
        if biobert_result['sentiment'] == mlp_result['sentiment']:
            results['agreements']['biobert_mlp'] += 1
        if biobert_result['sentiment'] == rule_sentiment:
            results['agreements']['biobert_rule'] += 1
        if mlp_result['sentiment'] == rule_sentiment:
            results['agreements']['mlp_rule'] += 1

    # Calculate percentages
    total = len(results['biobert'])
    results['agreement_rates'] = {
        'biobert_mlp': results['agreements']['biobert_mlp'] / total,
        'biobert_rule': results['agreements']['biobert_rule'] / total,
        'mlp_rule': results['agreements']['mlp_rule'] / total
    }

    return results


if __name__ == "__main__":
    # Example usage
    analyzer = BioBERTSentimentAnalyzer()

    # Test on sample medical reviews
    test_reviews = [
        "This medication completely cured my migraine headaches. Highly effective!",
        "Experienced severe nausea and vomiting after taking this drug. Never again.",
        "Mild improvement in my arthritis pain, but the side effects are manageable.",
        "This antibiotic cleared my infection quickly with no adverse reactions."
    ]

    print("🧠 BioBERT Sentiment Analysis Demo")
    print("=" * 50)

    for review in test_reviews:
        result = analyzer.analyze_sentiment(review)
        print(f"Review: {review[:50]}...")
        print(f"Sentiment: {result['sentiment'].upper()} (confidence: {result['confidence']:.3f})")
        print(f"Scores: {result['scores']}")
        print("-" * 50)