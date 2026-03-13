"""
Temporary test file for data loader modules.

Tests loading of WebMD reviews, Indian substitutes, and medicine reference dataset.
This file verifies that all data loading functions work correctly.
"""

from backend.modules.data_loader import (
    load_webmd_reviews,
    load_indian_substitutes,
    load_medicine_dataset
)


if __name__ == "__main__":
    # Load WebMD reviews
    print("Loading WebMD reviews...")
    reviews_df = load_webmd_reviews("backend/data/DS1_WebMD_Reviews.xlsx")
    print(f"[OK] Total reviews loaded: {len(reviews_df)}")
    print(f"     Columns: {list(reviews_df.columns)}")
    
    # Load Indian substitutes
    print("\nLoading Indian substitutes...")
    substitutes_dict = load_indian_substitutes("backend/data/DS2_IndianSubstitutes.xlsx")
    print(f"[OK] Total substitute mappings loaded: {len(substitutes_dict)}")
    if substitutes_dict:
        sample_brand = next(iter(substitutes_dict.keys()))
        print(f"     Sample mapping: {sample_brand} -> {substitutes_dict[sample_brand]}")
    
    # Load medicine reference dataset
    print("\nLoading medicine reference dataset...")
    print("     (This may take a minute - the dataset is large)")
    medicine_dict = load_medicine_dataset("backend/data/DS3_MedicineDataset.csv")
    print(f"[OK] Total medicines in reference: {len(medicine_dict)}")
    
    # Show sample medicine with side effects
    if medicine_dict:
        print("\n--- Sample Drug ---")
        sample_drug = next(iter(medicine_dict.keys()))
        sample_effects = medicine_dict[sample_drug]
        print(f"Drug: {sample_drug}")
        print(f"Number of side effects: {len(sample_effects)}")
        print(f"First 5 side effects: {list(sample_effects)[:5]}")
    
    print("\n[OK] All data loaded successfully!")
