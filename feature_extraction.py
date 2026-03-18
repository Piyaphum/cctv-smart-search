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
from deepface import DeepFace
from sklearn.cluster import KMeans


def extract_embedding(image_pil, reid_model, transform):
    """Extract feature embedding from image"""
    img_tensor = transform(image_pil).unsqueeze(0)
    with torch.no_grad():
        feature = reid_model(img_tensor).flatten().cpu().numpy()
    return feature


def get_text_embedding(text, clip_model, clip_processor):
    """Get embedding from text description"""
    inputs = clip_processor(text=[text], return_tensors="pt", padding=True)
    with torch.no_grad():
        text_outputs = clip_model.text_model(**inputs)
        last_hidden = text_outputs.last_hidden_state
        cls_token = last_hidden[:, 0, :]
        if hasattr(clip_model, 'text_projection'):
            text_embeds = clip_model.text_projection(cls_token)
        else:
            text_embeds = cls_token
    normalized = F.normalize(text_embeds, p=2, dim=-1)
    result = normalized.squeeze().cpu().detach().numpy()
    return result if result.ndim == 1 else result.flatten()


def get_image_embedding_clip(image_pil, clip_model, clip_processor):
    """Get image embedding from CLIP"""
    inputs = clip_processor(images=image_pil, return_tensors="pt")
    with torch.no_grad():
        vision_outputs = clip_model.vision_model(**inputs)
        last_hidden = vision_outputs.last_hidden_state
        cls_token = last_hidden[:, 0, :]
        if hasattr(clip_model, 'visual_projection'):
            image_embeds = clip_model.visual_projection(cls_token)
        else:
            image_embeds = cls_token
    normalized = F.normalize(image_embeds, p=2, dim=-1)
    result = normalized.squeeze().cpu().detach().numpy()
    return result if result.ndim == 1 else result.flatten()


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


def extract_colors_from_text(text):
    """Extract color names from search text"""
    colors = {
        'white': ['white', 'ขาว'],
        'black': ['black', 'ดำ'],
        'red': ['red', 'แดง'],
        'blue': ['blue', 'น้ำเงิน', 'ฟ้า'],
        'green': ['green', 'เขียว'],
        'yellow': ['yellow', 'เหลือง'],
        'orange': ['orange', 'ส้ม'],
        'purple': ['purple', 'ม่วง'],
        'pink': ['pink', 'ชมพู'],
        'brown': ['brown', 'น้ำตาล', 'กาแฟ'],
        'gray': ['gray', 'grey', 'เทา'],
    }
    
    found_colors = []
    text_lower = text.lower()
    for color_name, variations in colors.items():
        for variation in variations:
            if variation in text_lower:
                found_colors.append(color_name)
                break
    return found_colors


def detect_gender(image_pil):
    """Detect gender from image"""
    try:
        img_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        result = DeepFace.analyze(img_cv, actions=['gender'], enforce_detection=False)
        
        if isinstance(result, list) and len(result) > 0:
            gender = result[0].get('gender', {})
            if isinstance(gender, dict):
                dominant_gender = max(gender.items(), key=lambda x: x[1])[0]
                return dominant_gender
        return 'Unknown'
    except:
        return 'Unknown'


def extract_gender_from_text(text):
    """Extract gender from search text"""
    text_lower = text.lower()
    female_keywords = ['woman', 'girl', 'female', 'lady', 'ผู้หญิง', 'หญิง', 'สาว']
    male_keywords = ['man', 'boy', 'male', 'gentleman', 'ผู้ชาย', 'ชาย', 'หนุ่ม']
    
    if any(keyword in text_lower for keyword in female_keywords):
        return 'Female'
    if any(keyword in text_lower for keyword in male_keywords):
        return 'Male'
    return None
