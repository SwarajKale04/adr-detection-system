"""
Check sample reviews to understand the data.
"""

from backend.modules.data_loader import load_webmd_reviews
from backend.modules.drug_mapper import normalize_drug_name

reviews_df = load_webmd_reviews('backend/data/DS1_WebMD_Reviews.xlsx')
print("First 3 reviews:")
for i in range(3):
    drug = reviews_df.iloc[i]['drug_name']
    normalized = normalize_drug_name(drug)
    text = reviews_df.iloc[i]['text'][:150]
    print(f"Drug: {drug}")
    print(f"Normalized: {normalized}")
    print(f"Text: {text}...")
    print()
