"""
Analytics Module.

Generates summary analytics and statistics from ADR detection results.
Provides aggregated insights for dashboard visualization.
"""

import json
from collections import Counter


def load_adr_results(filepath: str) -> list:
    """
    Load ADR results from JSON file.
    
    Args:
        filepath (str): Path to adr_results.json
        
    Returns:
        list: List of ADR entry dictionaries
    """
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_analytics(adr_results: list) -> dict:
    """
    Generate comprehensive analytics from ADR results.
    
    Args:
        adr_results (list): List of ADR entries
        
    Returns:
        dict: Analytics dictionary containing:
              - total_adr_entries
              - unique_drugs
              - unique_symptoms
              - top_drugs (dict with counts)
              - top_symptoms (dict with counts)
              - average_confidence
              - sentiment_breakdown
    """
    if not adr_results:
        return {
            "total_adr_entries": 0,
            "unique_drugs": 0,
            "unique_symptoms": 0,
            "top_drugs": {},
            "top_symptoms": {},
            "average_confidence": 0.0,
            "sentiment_breakdown": {}
        }
    
    # Count drugs and symptoms
    drug_counts = Counter(entry["indian_drug"] for entry in adr_results)
    symptom_counts = Counter(entry["symptom"] for entry in adr_results)
    sentiment_counts = Counter(entry["sentiment"] for entry in adr_results)
    
    # Calculate average confidence
    total_confidence = sum(entry.get("confidence", 0.5) for entry in adr_results)
    average_confidence = round(total_confidence / len(adr_results), 2)
    
    # Build analytics dictionary
    analytics = {
        "total_adr_entries": len(adr_results),
        "unique_drugs": len(drug_counts),
        "unique_symptoms": len(symptom_counts),
        "top_drugs": dict(drug_counts.most_common(10)),
        "top_symptoms": dict(symptom_counts.most_common(10)),
        "average_confidence": average_confidence,
        "sentiment_breakdown": dict(sentiment_counts)
    }
    
    return analytics


def print_analytics(analytics: dict) -> None:
    """
    Print analytics in readable format.
    
    Args:
        analytics (dict): Analytics dictionary from generate_analytics()
    """
    print("\n" + "="*60)
    print("ADR ANALYTICS SUMMARY")
    print("="*60)
    
    print(f"\nTotal ADR Entries: {analytics['total_adr_entries']}")
    print(f"Unique Drugs: {analytics['unique_drugs']}")
    print(f"Unique Symptoms: {analytics['unique_symptoms']}")
    print(f"Average Confidence: {analytics['average_confidence']}")
    
    print("\nSentiment Breakdown:")
    for sentiment, count in analytics.get("sentiment_breakdown", {}).items():
        percentage = round((count / analytics['total_adr_entries']) * 100, 1)
        print(f"  {sentiment.capitalize()}: {count} ({percentage}%)")
    
    print("\nTop 10 Drugs by ADR Count:")
    for i, (drug, count) in enumerate(analytics["top_drugs"].items(), 1):
        percentage = round((count / analytics['total_adr_entries']) * 100, 1)
        print(f"  {i:2d}. {drug:30s} : {count:5d} ({percentage:5.1f}%)")
    
    print("\nTop 10 Symptoms by Occurrence:")
    for i, (symptom, count) in enumerate(analytics["top_symptoms"].items(), 1):
        percentage = round((count / analytics['total_adr_entries']) * 100, 1)
        print(f"  {i:2d}. {symptom:30s} : {count:5d} ({percentage:5.1f}%)")
    
    print("\n" + "="*60)


def main():
    """
    Main function to load and analyze ADR results.
    """
    filepath = "backend/output/adr_results.json"
    
    print("Loading ADR results...")
    adr_results = load_adr_results(filepath)
    
    print("Generating analytics...")
    analytics = generate_analytics(adr_results)
    
    print_analytics(analytics)
    
    return analytics


if __name__ == "__main__":
    analytics = main()
