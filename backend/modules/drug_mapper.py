"""
Drug Mapper Module.

Maps drug names from WebMD reviews to Indian medicine equivalents.
Handles drug name normalization and cross-reference lookup against
the substitutes dataset.

Key responsibilities:
- Normalize drug names (lowercase, remove dosage, remove form words)
- Map drug names to Indian generic names
- Handle unmapped drugs with normalized name fallback
"""

import re


def normalize_drug_name(drug_name: str) -> str:
    """
    Normalize drug name for consistent comparison and mapping.
    
    Operations:
    - Convert to lowercase
    - Remove dosage values (e.g., 500mg, 120 mg, 10ml)
    - Remove form words (tablet, syrup, capsule, injection, drops)
    - Normalize multiple spaces to single space
    - Strip leading/trailing whitespace
    
    Args:
        drug_name (str): Raw drug name
        
    Returns:
        str: Normalized drug name
    """
    # Convert to lowercase
    drug_name = str(drug_name).lower()
    
    # Remove dosage numbers (e.g., 120mg, 500 mg, 10ml, 5g)
    drug_name = re.sub(r'\d+\s*mg\b', '', drug_name)
    drug_name = re.sub(r'\d+\s*ml\b', '', drug_name)
    drug_name = re.sub(r'\d+\s*g\b', '', drug_name)
    
    # Remove form words
    form_words = ['tablet', 'syrup', 'capsule', 'injection', 'drops']
    for word in form_words:
        drug_name = re.sub(r'\b' + word + r'\b', '', drug_name, flags=re.IGNORECASE)
    
    # Normalize multiple spaces to single space
    drug_name = re.sub(r'\s+', ' ', drug_name)
    
    # Strip leading/trailing whitespace
    drug_name = drug_name.strip()
    
    return drug_name


def map_to_indian_generic(drug_name: str, substitutes_dict: dict) -> str:
    """
    Map drug name to Indian generic name using substitutes dictionary.
    
    Normalizes the input drug name, then looks it up in the substitutes
    dictionary. If found, returns the mapped generic name. Otherwise,
    returns the normalized drug name itself.
    
    Args:
        drug_name (str): Drug name to map
        substitutes_dict (dict): Dictionary mapping brand names to generic names
        
    Returns:
        str: Indian generic name or normalized drug name if not in mapping
    """
    normalized_name = normalize_drug_name(drug_name)
    
    if normalized_name in substitutes_dict:
        return substitutes_dict[normalized_name]
    
    return normalized_name
