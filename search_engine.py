"""
Search Engine - Core search logic
"""
import cv2
import numpy as np
from scipy.spatial.distance import cosine


def calculate_similarity(emb1, emb2):
    """
    Calculate similarity between two embeddings
    Returns: similarity score (0-1, higher is more similar)
    """
    if emb1.ndim > 1:
        emb1 = emb1.flatten()
    if emb2.ndim > 1:
        emb2 = emb2.flatten()
    
    similarity = 1 - cosine(emb1, emb2)
    return max(0, min(1, similarity))


def match_person(
    detected_person_embeddings,
    detected_person_hist,
    target_embeddings,
    target_hist_full,
    target_hist_top,
    threshold=0.70,
    shirt_strictness=0.6
):
    """
    Check if detected person matches target
    
    Args:
        detected_person_embeddings: list of embeddings
        detected_person_hist: histogram
        target_embeddings: list of target embeddings
        target_hist_full: target full histogram
        target_hist_top: target top histogram
        threshold: similarity threshold
        shirt_strictness: weight for color matching (0-1)
    
    Returns:
        (is_match, best_similarity, best_color_score)
    """
    if not detected_person_embeddings or not target_embeddings:
        return False, 0, 0
    
    # Find best embedding match
    best_similarity = 0
    for det_emb in detected_person_embeddings:
        for target_emb in target_embeddings:
            sim = calculate_similarity(det_emb, target_emb)
            best_similarity = max(best_similarity, sim)
    
    # Check histogram similarity
    hist_similarity_full = cv2.compareHist(
        detected_person_hist.reshape(1, -1),
        target_hist_full[0].reshape(1, -1),
        cv2.HISTCMP_COSINE
    ) if len(target_hist_full) > 0 else 0.5
    
    hist_similarity = max(hist_similarity_full, 0.3)
    
    # Combined score
    combined_score = best_similarity * (1 - shirt_strictness) + hist_similarity * shirt_strictness
    
    # Check if meets threshold
    is_match = combined_score >= threshold
    
    return is_match, best_similarity, hist_similarity


def batch_match_targets(
    detected_embeddings,
    detected_hist,
    targets,
    threshold=0.70,
    shirt_strictness=0.6
):
    """
    Match detected person against multiple targets
    
    Returns:
        list of (target_idx, similarity, color_score)
    """
    matches = []
    
    for target_idx, target in enumerate(targets):
        is_match, sim, color_sim = match_person(
            detected_embeddings,
            detected_hist,
            target['embeddings'],
            target['hists_full'],
            target['hists_top'],
            threshold,
            shirt_strictness
        )
        
        if is_match:
            matches.append({
                'target_idx': target_idx,
                'target_name': target['name'],
                'similarity': sim,
                'color_score': color_sim
            })
    
    return matches
