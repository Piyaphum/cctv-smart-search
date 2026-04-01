"""
Target Profile Management
"""
import torch
from PIL import Image
from sklearn.cluster import KMeans
from feature_extraction import (
    extract_embedding,
    get_part_histogram,
    get_dominant_color_name,
    detect_gender
)


def generate_target_data(image_file, detector, reid_model, base_tf, aug_tf, n_aug=3):
    """
    Generate embeddings and features for target image
    
    Args:
        image_file: uploaded image file
        detector: YOLO detector
        reid_model: ReID model
        base_tf: base transform
        aug_tf: augmentation transform
        n_aug: number of augmentations
    
    Returns:
        dict with embeddings, histograms, color, gender
    """
    img = Image.open(image_file).convert('RGB')
    
    # Auto-crop if person detected
    import cv2
    import numpy as np
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    results = detector(img_cv, classes=0, verbose=False)
    if len(results[0].boxes) > 0:
        box = results[0].boxes[0].xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = map(int, box)
        img = img.crop((x1, y1, x2, y2))
    
    embeddings = []
    hists_full = []
    hists_top = []
    
    def process_image(pil_img):
        embeddings.append(extract_embedding(pil_img, reid_model, base_tf))
        hists_full.append(get_part_histogram(pil_img, 'full'))
        hists_top.append(get_part_histogram(pil_img, 'top'))
    
    process_image(img)
    for _ in range(n_aug):
        aug_img = aug_tf(img)
        process_image(aug_img)
    
    # Extract color and gender
    color = get_dominant_color_name(img)
    gender = detect_gender(img)
    
    return {
        "name": image_file.name,
        "image": img,
        "embeddings": embeddings,
        "hists_full": hists_full,
        "hists_top": hists_top,
        "color": color,
        "gender": gender,
        "type": "image"
    }


def prepare_targets_for_search(selected_targets):
    """
    Prepare target data for search operation
    
    Args:
        selected_targets: list of target data dicts
    
    Returns:
        prepared targets list
    """
    return selected_targets
