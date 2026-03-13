"""
Quick test of the pipeline using mock data for faster testing.
"""

import pandas as pd
from backend.modules.data_loader import load_webmd_reviews, load_indian_substitutes
from backend.modules.pipeline import run_pipeline

# Load real reviews and substitutes
print("Loading reviews and substitutes...")
reviews_df = load_webmd_reviews('backend/data/DS1_WebMD_Reviews.xlsx')
substitutes_dict = load_indian_substitutes('backend/data/DS2_IndianSubstitutes.xlsx')

# Create mock medicine reference (small sample for testing)
medicine_reference = {
    'aspirin': {'headache', 'stomachache', 'nausea'},
    'ibuprofen': {'dizziness', 'nausea', 'rash'},
    'augmentin 625 duo': {'vomiting', 'nausea', 'diarrhea'},
    'lisinopril': {'dizziness', 'cough', 'headache'},
    'metformin': {'nausea', 'diarrhea', 'vomiting'},
    'zosert': {'anxiety', 'anger', 'depression'}
}

# Run pipeline on first 100 reviews
print("Running pipeline on first 100 reviews...")
results = run_pipeline(reviews_df, substitutes_dict, medicine_reference, limit=100)

print(f"Total ADR entries detected: {len(results)}")
if results:
    print("\nFirst 5 ADR entries:")
    for entry in results[:5]:
        print(f"  Drug: {entry['original_drug']} -> {entry['indian_drug']}")
        print(f"  Symptom: {entry['symptom']}")
        print(f"  Sentiment: {entry['sentiment']}")
        print(f"  Validated: {entry['validated']}")
        print()
