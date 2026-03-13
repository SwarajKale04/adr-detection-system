"""
Negation Handler Module.

Filters out detected side effects that are negated in the review text.

Uses simple rule-based negation detection by looking for negation words
(no, not, never, without, didn't, did not) appearing 3-4 words before
each detected effect.
"""


def filter_negated_effects(
    review_text: str,
    detected_effects: list
) -> list:
    """
    Filter out side effects that are negated in the review text.
    
    For each detected effect, checks if negation words appear 3-4 words
    before it in the review. If a negation is found, the effect is removed
    from the result list.
    
    Negation words: no, not, never, without, didn't, did not
    
    Args:
        review_text (str): Review text (already lowercase)
        detected_effects (list): List of detected side effects from symptom_detector
        
    Returns:
        list: Filtered list with negated effects removed
    """
    negation_words = ["no", "not", "never", "without", "didn't", "did not"]
    
    filtered_effects = []
    
    for effect in detected_effects:
        is_negated = False
        
        # Find the effect in the review text
        effect_index = review_text.find(effect)
        
        if effect_index != -1:
            # Extract text before the effect (look back ~30 chars for 3-4 words context)
            context_start = max(0, effect_index - 30)
            context_text = review_text[context_start:effect_index]
            context_words = context_text.split()
            
            # Check if any negation word appears in the context
            for neg_word in negation_words:
                if neg_word in context_words:
                    is_negated = True
                    break
        
        # Add to result if not negated
        if not is_negated:
            filtered_effects.append(effect)
    
    return filtered_effects
