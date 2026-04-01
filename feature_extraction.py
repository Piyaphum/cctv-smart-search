"""
Feature Extraction Functions
- Extract embeddings
- Extract color information
- Gender detection
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.cluster import KMeans


def extract_embedding(image_pil, reid_model, transform):
    """Extract feature embedding from image"""
    img_tensor = transform(image_pil).unsqueeze(0)
    with torch.no_grad():
        feature = reid_model(img_tensor).flatten().cpu().numpy()
    return feature





def get_part_histogram(image_pil, part='full'):
    """Extract color histogram from different body parts"""
    img_np = np.array(image_pil)
    h, w, _ = img_np.shape
    
    if part == 'top':
        img_crop = img_np[int(h*0.25):int(h*0.60), :]
    elif part == 'bottom':
        img_crop = img_np[int(h*0.60):int(h*0.95), :]
    else:
        img_crop = img_np
    
    img_hsv = cv2.cvtColor(img_crop, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([img_hsv], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def closest_color(rgb_tuple):
    """Convert RGB to color name"""
    try:
        r, g, b = int(rgb_tuple[0]), int(rgb_tuple[1]), int(rgb_tuple[2])
        if r > 200 and g > 200 and b > 200: return "white"
        if r < 50 and g < 50 and b < 50: return "black"
        if r > g + 30 and r > b + 30: return "red"
        if g > r + 30 and g > b + 30: return "green"
        if b > r + 30 and b > g + 30: return "blue"
        if r > 150 and g > 70 and b < 50: return "orange"
        if r > 150 and g > 150 and b < 50: return "yellow"
        if r > 100 and g < 100 and b > 100: return "purple"
        if r > 150 and g < 150 and b > 150: return "pink"
        if r < 100 and g > 100 and b > 100: return "cyan"
        if r > 100 and g > 100 and b < 100: return "brown"
        if (r + g + b) // 3 > 150: return "gray"
        return "unknown"
    except:
        return "unknown"


def get_dominant_color_name(image_pil):
    """Extract dominant clothing color"""
    try:
        img_np = np.array(image_pil)
        h, w, c = img_np.shape
        top_start = int(h * 0.25)
        top_end = int(h * 0.65)
        top_crop = img_np[top_start:top_end, :]
        
        if top_crop.size == 0:
            return "Unknown"
        
        pixels = top_crop.reshape(-1, 3).astype(np.float32)
        brightness = np.mean(pixels, axis=1)
        valid_mask = (brightness > 20) & (brightness < 230)
        valid_pixels = pixels[valid_mask]
        
        if len(valid_pixels) < 10:
            valid_pixels = pixels
        
        n_clusters = min(3, len(np.unique(valid_pixels, axis=0)))
        kmeans = KMeans(n_clusters=n_clusters, n_init=5, random_state=42)
        kmeans.fit(valid_pixels)
        
        counts = np.bincount(kmeans.labels_)
        dominant_idx = np.argmax(counts)
        dominant = kmeans.cluster_centers_[dominant_idx]
        
        color_name = closest_color([int(dominant[0]), int(dominant[1]), int(dominant[2])])
        return color_name.capitalize()
    except:
        return "Unknown"





def detect_gender(image_pil):
    """Detect gender from image (DeepFace removed)"""
    return 'Unknown'