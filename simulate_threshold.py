import numpy as np

def run_threshold_test():
    # Realistic similarity scores for the SAME person (Target)
    # They range from 0.61 to 0.95 due to different angles, lighting, occlusion
    pos_scores = [0.95, 0.88, 0.85, 0.82, 0.79, 0.76, 0.72, 0.71, 0.68, 0.66, 0.65, 0.61]
    
    # Realistic similarity scores for DIFFERENT people (Strangers)
    # Some strangers wearing similar clothes might score high (e.g. 0.78), but most are lower
    neg_scores = [0.78, 0.74, 0.69, 0.67, 0.65, 0.62, 0.58, 0.55, 0.50, 0.48, 0.45, 0.40]

    thresholds = [0.60, 0.70, 0.80, 0.90]

    for threshold in thresholds:
        print("\n" + "=" * 50)
        print(f"Testing with Threshold: {threshold*100:.0f}% ({threshold:.2f})")
        print("=" * 50)
        
        tp = sum(1 for s in pos_scores if s >= threshold)
        fn = len(pos_scores) - tp
        fp = sum(1 for s in neg_scores if s >= threshold)
        tn = len(neg_scores) - fp
        
        accuracy = (tp + tn) / (len(pos_scores) + len(neg_scores))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"  True Positives (Found Target)     : {tp}/{len(pos_scores)}")
        print(f"  False Positives (Wrong Person)    : {fp}/{len(neg_scores)}")
        print(f"  True Negatives (Ignored Stranger) : {tn}/{len(neg_scores)}")
        print(f"  False Negatives (Missed Target)   : {fn}/{len(pos_scores)}")
        print(f"  -----------------------------------")
        print(f"  Precision : {precision*100:.1f}%  (When it says 'Match', how often is it correct?)")
        print(f"  Recall    : {recall*100:.1f}%  (Out of all real targets, how many did it catch?)")
        print(f"  F1-Score  : {f1_score*100:.1f}%")
        print(f"  Accuracy  : {accuracy*100:.1f}%")

if __name__ == "__main__":
    run_threshold_test()
