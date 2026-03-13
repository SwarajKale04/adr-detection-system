"""
Data Loader Module.

Responsible for loading and preprocessing data from multiple sources:
- WebMD reviews dataset (Excel)
- Indian medicine substitutes mapping (Excel)
- Medicine reference dataset (CSV)

Provides utilities to read, parse, and validate input data formats.
"""

import pandas as pd
from .drug_mapper import normalize_drug_name


def load_webmd_reviews(filepath: str) -> pd.DataFrame:
    """
    Load WebMD reviews from Excel file.
    
    Loads the Excel file, keeps only drug_name and text columns,
    drops rows with empty reviews, and converts review text to lowercase.
    
    Args:
        filepath (str): Path to DS1_WebMD_Reviews.xlsx file
        
    Returns:
        pd.DataFrame: DataFrame with columns [drug_name, text] where
                     reviews are lowercase and non-empty
    """
    df = pd.read_excel(filepath)
    df = df[['drug_name', 'text']]
    df = df.dropna(subset=['text'])
    df = df[df['text'].astype(str).str.strip() != '']
    df['text'] = df['text'].astype(str).str.lower()
    
    return df


def load_indian_substitutes(filepath: str) -> dict:
    """
    Load Indian medicine substitutes mapping from Excel file.
    
    Creates a dictionary mapping brand name to generic name.
    Uses normalize_drug_name() for consistent normalization across the system.
    
    Args:
        filepath (str): Path to DS2_IndianSubstitutes.xlsx file
        
    Returns:
        dict: Dictionary with format {brand_name_normalized: generic_name_normalized}
    """
    df = pd.read_excel(filepath)
    substitutes_dict = {}
    
    for _, row in df.iterrows():
        brand_name = normalize_drug_name(row.iloc[0])
        generic_name = normalize_drug_name(row.iloc[1])
        substitutes_dict[brand_name] = generic_name
    
    return substitutes_dict


def load_medicine_dataset(filepath: str) -> dict:
    """
    Load medicine reference dataset from CSV file.
    
    Normalizes drug names using normalize_drug_name() from drug_mapper.
    Extracts all sideEffect columns dynamically, ignores empty cells,
    and stores side effects in a set. Merges duplicate drug rows by
    updating the side effect set.
    
    Args:
        filepath (str): Path to DS3_MedicineDataset.csv file
        
    Returns:
        dict: Dictionary with format {normalized_drug_name: set(side_effects)}
    """
    print("  Reading CSV file...")
    df = pd.read_csv(filepath, low_memory=False)
    
    # Precompute side effect columns once
    side_effect_cols = [col for col in df.columns if col.startswith('sideEffect')]
    
    print("  Processing medicines...")
    medicine_dict = {}
    
    for idx, row in df.iterrows():
        # Print progress every 10,000 rows
        if idx % 10000 == 0:
            print(f"    Processed {idx} medicines...", end='\r')
        
        # Normalize drug name using consistent normalization function
        drug_name = normalize_drug_name(row['name'])
        
        # Skip empty drug names
        if not drug_name:
            continue
        
        # Extract side effects
        side_effects = set()
        for col in side_effect_cols:
            effect = str(row[col]).strip().lower()
            if effect and effect != 'nan':
                side_effects.add(effect)
        
        # Merge duplicate drugs by updating side effect set
        if drug_name in medicine_dict:
            medicine_dict[drug_name].update(side_effects)
        else:
            medicine_dict[drug_name] = side_effects
    
    print()  # Clear progress line
    return medicine_dict
