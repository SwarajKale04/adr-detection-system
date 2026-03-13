from backend.modules.symptom_detector import detect_known_side_effects
from backend.modules.drug_mapper import normalize_drug_name
import pandas as pd
import os
import json
import argparse
from backend.modules.pipeline import run_pipeline
from backend.modules.data_loader import load_webmd_reviews, load_indian_substitutes, load_medicine_dataset
from backend.modules.analytics import load_adr_results, generate_analytics, print_analytics


def main(use_mlp_sentiment: bool = False):
    """
    Main ADR detection pipeline with optional MLP sentiment analysis.

    Args:
        use_mlp_sentiment (bool): Whether to use MLP model for sentiment analysis
    """
    print("🔍 ADR Detection System")
    print("=" * 50)

    if use_mlp_sentiment:
        print("🤖 Using MLP-based sentiment analysis")
    else:
        print("📋 Using rule-based sentiment analysis")

    # Load full medicine reference from dataset
    print("\n📚 Loading medicine reference from dataset...")
    medicine_reference = load_medicine_dataset("backend/data/DS3_MedicineDataset.csv")
    print(f"✅ Total drugs in medicine_reference: {len(medicine_reference)}")

    # Load reviews and substitutes
    print("\n📖 Loading reviews and substitutes...")
    reviews_df = load_webmd_reviews('backend/data/DS1_WebMD_Reviews.xlsx')
    substitutes_dict = load_indian_substitutes('backend/data/DS2_IndianSubstitutes.xlsx')
    print(f"✅ Loaded {len(reviews_df)} reviews from {len(reviews_df['drug_name'].unique())} drugs")

    # Run pipeline
    print(f"\n🚀 Running pipeline on all reviews...")
    adr_results, drug_sentiment = run_pipeline(
        reviews_df,
        substitutes_dict,
        medicine_reference,
        limit=999999,
        use_mlp_sentiment=use_mlp_sentiment
    )

    print(f"\n📊 Results Summary:")
    print(f"   • ADR entries detected: {len(adr_results)}")
    print(f"   • Drugs with sentiment tracking: {len(drug_sentiment)}")

    # Export results to JSON files
    output_dir = "backend/output"
    os.makedirs(output_dir, exist_ok=True)

    # ADR results
    output_path = os.path.join(output_dir, "adr_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(adr_results, f, indent=2, ensure_ascii=False)
    print(f"💾 ADR results saved to: {output_path}")

    # Drug sentiment summary
    sentiment_path = os.path.join(output_dir, "drug_sentiment_summary.json")
    with open(sentiment_path, "w", encoding="utf-8") as f:
        json.dump(drug_sentiment, f, indent=2, ensure_ascii=False)
    print(f"💾 Drug sentiment summary saved to: {sentiment_path}")

    # Generate and display analytics
    print("\n📈 Generating analytics...")
    analytics = generate_analytics(adr_results)
    print_analytics(analytics)

    if adr_results:
        print(f"\n🔍 Sample ADR entry: {adr_results[0]['symptom']} - {adr_results[0]['indian_drug']}")

    print("\n✅ Pipeline completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ADR Detection System')
    parser.add_argument('--mlp', action='store_true',
                       help='Use MLP model for sentiment analysis instead of rule-based')
    parser.add_argument('--train-mlp', action='store_true',
                       help='Train the MLP model first, then run pipeline')

    args = parser.parse_args()

    if args.train_mlp:
        print("🎯 Training MLP sentiment model...")
        from backend.modules.mlp_sentiment_analyzer import train_mlp_model
        train_results = train_mlp_model()
        print(f"✅ MLP training completed with accuracy: {train_results['accuracy']:.3f}")

    main(use_mlp_sentiment=args.mlp)
