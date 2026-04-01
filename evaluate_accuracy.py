import os
import cv2
import numpy as np
import torch
from PIL import Image
from scipy.spatial.distance import cosine
import sys

# Add current directory to path for imports
sys.path.append(os.getcwd())

from models import get_all_models
from feature_extraction import extract_embedding, get_part_histogram

def calculate_similarity(emb1, emb2):
    if emb1.ndim > 1:
        emb1 = emb1.flatten()
    if emb2.ndim > 1:
        emb2 = emb2.flatten()
    similarity = 1 - cosine(emb1, emb2)
    return max(0, min(1, similarity))

def run_evaluation():
    print("Loading models...")
    models = get_all_models()
    reid_model = models['reid_model']
    base_tf = models['base_transform']
    
    # Paths
    pos_dir = "detected_results/admin/suspicious"
    neg_dir = "detected_results/admin/mix"
    
    pos_files = [os.path.join(pos_dir, f) for f in os.listdir(pos_dir) if f.endswith('.jpg')][:10]
    neg_files = [os.path.join(neg_dir, f) for f in os.listdir(neg_dir) if f.endswith('.jpg')][:10]
    
    if len(pos_files) < 10 or len(neg_files) < 10:
        print(f"Error: Not enough sample files. Pos: {len(pos_files)}, Neg: {len(neg_files)}")
        return

    print(f"Processing 10 positive and 10 negative samples...")
    
    # Extract features for all samples
    def get_features(file_path):
        img = Image.open(file_path).convert('RGB')
        emb = extract_embedding(img, reid_model, base_tf)
        hist = get_part_histogram(img, 'full')
        return emb, hist

    pos_features = [get_features(f) for f in pos_files]
    neg_features = [get_features(f) for f in neg_files]
    
    # Use the first positive sample as the target
    target_emb, target_hist = pos_features[0]
    
    print("\nResults for Target: " + os.path.basename(pos_files[0]))
    print("-" * 50)
    
    shirt_strictness = 0.6
    
    # Pre-calculate scores since features are already extracted
    pos_scores = []
    print("Pre-calculating scores for Positive Samples:")
    for i, (emb, hist) in enumerate(pos_features):
        sim = calculate_similarity(emb, target_emb)
        hist_sim = 1 - cosine(hist.flatten(), target_hist.flatten())
        combined = sim * (1 - shirt_strictness) + hist_sim * shirt_strictness
        pos_scores.append(combined)

    neg_scores = []
    print("Pre-calculating scores for Negative Samples:")
    for i, (emb, hist) in enumerate(neg_features):
        sim = calculate_similarity(emb, target_emb)
        hist_sim = 1 - cosine(hist.flatten(), target_hist.flatten())
        combined = sim * (1 - shirt_strictness) + hist_sim * shirt_strictness
        neg_scores.append(combined)

    for threshold in [0.60, 0.70, 0.80, 0.90]:
        print("\n" + "=" * 50)
        print(f"Testing with Threshold: {threshold:.2f} ({(threshold*100):.0f}%)")
        print("=" * 50)
        
        tp, fp, tn, fn = 0, 0, 0, 0
        
        # Test positive samples (should be matches)
        print("Positive Samples (Same Person):")
        for i, combined in enumerate(pos_scores):
            is_match = combined >= threshold
            status = "CORRECT (Match)" if is_match else "WRONG (Miss)"
            if is_match: tp += 1
            else: fn += 1
            print(f"  {i+1}. Score: {combined:.4f} -> {status}")

        # Test negative samples (should be non-matches)
        print("\nNegative Samples (Different People):")
        for i, combined in enumerate(neg_scores):
            is_match = combined >= threshold
            status = "WRONG (Fake Match)" if is_match else "CORRECT (No Match)"
            if is_match: fp += 1
            else: tn += 1
            print(f"  {i+1}. Score: {combined:.4f} -> {status}")

        # Final Statistics
        accuracy = (tp + tn) / 20
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print("-" * 50)
        print(f"Summary for Threshold {threshold:.2f}:")
        print(f"  True Positives (Correct Matches): {tp}/10")
        print(f"  False Positives (Fake Matches): {fp}/10")
        print(f"  True Negatives (Correct Rejects): {tn}/10")
        print(f"  False Negatives (Missed Targets): {fn}/10")
        print(f"  Precision: {precision*100:.1f}%")
        print(f"  Recall: {recall*100:.1f}%")
        print(f"  F1-Score: {f1_score*100:.1f}%")
        print(f"  Overall Accuracy: {accuracy*100:.1f}%")

if __name__ == "__main__":
    run_evaluation()
