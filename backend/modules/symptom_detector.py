"""
Symptom Detector Module.

Detects adverse drug reactions (ADR) by matching known side effects
from the medicine reference dictionary against review text.

Uses simple substring matching to identify side effects mentioned
in drug reviews.
"""


def detect_known_side_effects(
    drug_name: str,
    review_text: str,
    medicine_reference: dict
) -> list:
    """
    Detect adverse drug reactions by matching known side effects.
    
    Matches known side effects for a specific drug against review text.
    Only detects side effects that are known for that drug from the
    medicine reference dictionary.
    
    Args:
        drug_name (str): Normalized drug name (already lowercase, no dosage/form)
        review_text (str): Review text (already lowercase)
        medicine_reference (dict): Dictionary mapping normalized drug names to
                                  sets of known side effects.
                                  Format: {drug_name: set(side_effects), ...}
        
    Returns:
        list: List of detected side effects found in the review text.
              Empty list if drug not found or no matches.
    """
    # If drug not in reference, return empty list
    if drug_name not in medicine_reference:
        return []
    
    # Get known side effects for this drug
    known_effects = medicine_reference[drug_name]
    
    # Find which known effects appear in the review text
    detected_effects = []
    for effect in known_effects:
        if effect in review_text:
            detected_effects.append(effect)
    
    return detected_effects
