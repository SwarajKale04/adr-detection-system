"""
Sentiment Analyzer Module.

Classifies review sentiment using simple rule-based keyword detection,
optimized for ADR (adverse drug reaction) context.

Uses keyword matching to classify reviews as negative, positive, or neutral.
"""


def analyze_sentiment(review_text: str) -> str:
    """
    Classify review sentiment using rule-based keyword detection.
    
    Checks for the presence of predefined sentiment keywords in the review text.
    Classification prioritizes negative keywords over positive ones.
    
    Keyword lists:
    - negative: severe, terrible, horrible, worst, bad, painful, vomiting, nausea,
                headache, dizziness, rash, anxiety, caused, made me, suffered, side effect
    - positive: good, great, effective, helped, improved, better, relieved, worked
    - neutral: okay, fine, average
    
    Classification rules:
    1. If any negative keyword appears → "negative"
    2. Else if any positive keyword appears → "positive"
    3. Else → "neutral"
    
    Args:
        review_text (str): Review text (already lowercase)
        
    Returns:
        str: Sentiment classification: "negative", "positive", or "neutral"
    """
    negative_keywords = [
        "severe", "terrible", "horrible", "worst",
        "bad", "painful", "vomiting", "nausea",
        "headache", "dizziness", "rash", "anxiety",
        "caused", "made me", "suffered", "side effect"
    ]
    
    positive_keywords = [
        "good", "great", "effective", "helped",
        "improved", "better", "relieved", "worked"
    ]
    
    neutral_keywords = [
        "okay", "fine", "average"
    ]
    
    # Check for negative keywords
    for keyword in negative_keywords:
        if keyword in review_text:
            return "negative"
    
    # Check for positive keywords
    for keyword in positive_keywords:
        if keyword in review_text:
            return "positive"
    
    # Check for neutral keywords (optional)
    for keyword in neutral_keywords:
        if keyword in review_text:
            return "neutral"
    
    # Default to neutral if no keywords match
    return "neutral"
