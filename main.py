from backend.modules.symptom_detector import detect_known_side_effects
from backend.modules.drug_mapper import normalize_drug_name
import pandas as pd
import os
import json
from backend.modules.pipeline import run_pipeline
from backend.modules.data_loader import load_webmd_reviews, load_indian_substitutes, load_medicine_dataset
from backend.modules.analytics import load_adr_results, generate_analytics, print_analytics


if __name__ == "__main__":
    # Load full medicine reference from dataset
    print("Loading medicine reference from dataset...")
    medicine_reference = load_medicine_dataset("backend/data/DS3_MedicineDataset.csv")
    print(f"Total drugs in medicine_reference: {len(medicine_reference)}")

    # now run full pipeline using real reviews and substitutes
    print("Loading reviews and substitutes...")
    reviews_df = load_webmd_reviews('backend/data/DS1_WebMD_Reviews.xlsx')
    substitutes_dict = load_indian_substitutes('backend/data/DS2_IndianSubstitutes.xlsx')
    print("Running pipeline on all reviews...")
    adr_results, drug_sentiment = run_pipeline(reviews_df, substitutes_dict, medicine_reference, limit=999999)

    print(f"\nTotal ADR entries detected: {len(adr_results)}")
    print(f"Total drugs with sentiment tracking: {len(drug_sentiment)}")

    # export results to JSON file
    output_dir = "backend/output"
    os.makedirs(output_dir, exist_ok=True)
    
    # ADR results
    output_path = os.path.join(output_dir, "adr_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(adr_results, f, indent=2, ensure_ascii=False)
    print(f"ADR results saved to: {output_path}")
    
    # Drug sentiment summary
    sentiment_path = os.path.join(output_dir, "drug_sentiment_summary.json")
    with open(sentiment_path, "w", encoding="utf-8") as f:
        json.dump(drug_sentiment, f, indent=2, ensure_ascii=False)
    print(f"Drug sentiment summary saved to: {sentiment_path}")
    
    # Generate and display analytics
    print("\nGenerating analytics...")
    analytics = generate_analytics(adr_results)
    print_analytics(analytics)
    print(f"Sample entry: {adr_results[0] if adr_results else 'No results'}")
