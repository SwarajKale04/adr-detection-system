"""
Pipeline Module.

Orchestrates the full ADR (Adverse Drug Reaction) detection pipeline
using all previously implemented modules.

Supports both rule-based and machine learning approaches for sentiment analysis.
"""

from .drug_mapper import normalize_drug_name, map_to_indian_generic
from .symptom_detector import detect_known_side_effects
from .negation_handler import filter_negated_effects
from .sentiment_analyzer import analyze_sentiment
from .mlp_sentiment_analyzer import MLPSentimentAnalyzer


def run_pipeline(
    reviews_df,
    substitutes_dict,
    medicine_reference,
    limit: int = 5000,
    use_mlp_sentiment: bool = False
) -> tuple[list, dict]:
    """
    Run the full ADR detection pipeline on drug reviews.

    Processes each review to:
    1. Normalize and map the drug name
    2. Detect side effects mentioned in the review
    3. Filter out negated effects
    4. Analyze sentiment (rule-based or MLP)
    5. Create ADR entries for negative reviews with detected effects

    Args:
        reviews_df: DataFrame with columns [drug_name, text]
        substitutes_dict: Dictionary mapping brand names to Indian generics
        medicine_reference: Dictionary mapping normalized drugs to side effects
        limit (int): Maximum number of reviews to process (default 5000)
        use_mlp_sentiment (bool): Whether to use MLP model for sentiment analysis

    Returns:
        tuple: (adr_results, drug_sentiment_summary)
               - adr_results: List of ADR entries
               - drug_sentiment_summary: Dict with sentiment tracking per drug
    """
    adr_results = []
    drug_sentiment = {}

    # Initialize MLP analyzer if needed
    mlp_analyzer = None
    if use_mlp_sentiment:
        mlp_analyzer = MLPSentimentAnalyzer()
        if not mlp_analyzer.load_model():
            print("⚠️  MLP model not found, falling back to rule-based sentiment analysis")
            use_mlp_sentiment = False

    # Process only first `limit` rows
    for idx, row in reviews_df.head(limit).iterrows():
        original_drug = str(row['drug_name']).lower()
        review_text = str(row['text']).lower()

        # Normalize drug name
        normalized_drug = normalize_drug_name(original_drug)

        # Map to Indian generic
        indian_drug = map_to_indian_generic(original_drug, substitutes_dict)

        # Analyze sentiment for all reviews (for sentiment tracking)
        if use_mlp_sentiment and mlp_analyzer:
            sentiment_result = mlp_analyzer.predict_sentiment(review_text)
            sentiment = sentiment_result['sentiment']
        else:
            sentiment = analyze_sentiment(review_text)

        # Update drug sentiment tracking
        if indian_drug not in drug_sentiment:
            drug_sentiment[indian_drug] = {
                "positive_count": 0,
                "negative_count": 0,
                "positive_examples": [],
                "negative_examples": []
            }

        if sentiment == "positive":
            drug_sentiment[indian_drug]["positive_count"] += 1
            if len(drug_sentiment[indian_drug]["positive_examples"]) < 3:
                drug_sentiment[indian_drug]["positive_examples"].append(review_text)
        elif sentiment == "negative":
            drug_sentiment[indian_drug]["negative_count"] += 1
            if len(drug_sentiment[indian_drug]["negative_examples"]) < 3:
                drug_sentiment[indian_drug]["negative_examples"].append(review_text)

        # Detect known side effects
        detected_effects = detect_known_side_effects(
            normalized_drug,
            review_text,
            medicine_reference
        )

        # Skip if no effects detected
        if not detected_effects:
            continue

        # Filter negated effects
        filtered_effects = filter_negated_effects(review_text, detected_effects)

        # Skip if all effects were negated
        if not filtered_effects:
            continue

        # Create ADR entries only if sentiment is negative
        if sentiment == "negative":
            for effect in filtered_effects:
                # calculate confidence score
                confidence = 0.5
                confidence += 0.2  # already negative sentiment
                if review_text.count(effect) > 1:
                    confidence += 0.2
                if len(review_text) > 120:
                    confidence += 0.1
                confidence = min(confidence, 1.0)
                confidence = round(confidence, 2)

                adr_entry = {
                    "original_drug": original_drug,
                    "indian_drug": indian_drug,
                    "symptom": effect,
                    "sentence": review_text,
                    "sentiment": sentiment,
                    "validated": True,
                    "confidence": confidence
                }
                adr_results.append(adr_entry)

    return adr_results, drug_sentiment
    adr_results = []
    drug_sentiment = {}
    
    # Process only first `limit` rows
    for idx, row in reviews_df.head(limit).iterrows():
        original_drug = str(row['drug_name']).lower()
        review_text = str(row['text']).lower()
        
        # Normalize drug name
        normalized_drug = normalize_drug_name(original_drug)
        
        # Map to Indian generic
        indian_drug = map_to_indian_generic(original_drug, substitutes_dict)
        
        # Analyze sentiment for all reviews (for sentiment tracking)
        sentiment = analyze_sentiment(review_text)
        
        # Update drug sentiment tracking
        if indian_drug not in drug_sentiment:
            drug_sentiment[indian_drug] = {
                "positive_count": 0,
                "negative_count": 0,
                "positive_examples": [],
                "negative_examples": []
            }
        
        if sentiment == "positive":
            drug_sentiment[indian_drug]["positive_count"] += 1
            if len(drug_sentiment[indian_drug]["positive_examples"]) < 3:
                drug_sentiment[indian_drug]["positive_examples"].append(review_text)
        elif sentiment == "negative":
            drug_sentiment[indian_drug]["negative_count"] += 1
            if len(drug_sentiment[indian_drug]["negative_examples"]) < 3:
                drug_sentiment[indian_drug]["negative_examples"].append(review_text)
        
        # Detect known side effects
        detected_effects = detect_known_side_effects(
            normalized_drug,
            review_text,
            medicine_reference
        )
        
        # Skip if no effects detected
        if not detected_effects:
            continue
        
        # Filter negated effects
        filtered_effects = filter_negated_effects(review_text, detected_effects)
        
        # Skip if all effects were negated
        if not filtered_effects:
            continue
        
        # Create ADR entries only if sentiment is negative
        if sentiment == "negative":
            for effect in filtered_effects:
                # calculate confidence score
                confidence = 0.5
                confidence += 0.2  # already negative sentiment
                if review_text.count(effect) > 1:
                    confidence += 0.2
                if len(review_text) > 120:
                    confidence += 0.1
                confidence = min(confidence, 1.0)
                confidence = round(confidence, 2)

                adr_entry = {
                    "original_drug": original_drug,
                    "indian_drug": indian_drug,
                    "symptom": effect,
                    "sentence": review_text,
                    "sentiment": sentiment,
                    "validated": True,
                    "confidence": confidence
                }
                adr_results.append(adr_entry)
    
    return adr_results, drug_sentiment
