"""
Video Processing Functions
"""
import cv2
import os
import numpy as np
from PIL import Image
from config import RESULT_DIR


def process_video_frame(frame, detector, reid_model, transform, target, threshold, shirt_strictness):
    """
    Process single video frame and detect matches
    Returns: list of (person_image, target_name, similarity, color, confidence)
    """
    results = detector(frame, classes=0, conf=0.3, verbose=False)
    
    # If no detections with 0.3 confidence, try with lower threshold
    if not results or len(results) == 0:
        results = detector(frame, classes=0, conf=0.15, verbose=False)
    
    detections = []
    
    for r in results:
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            continue
        
        for box in boxes:
            try:
                # Extract and convert coordinates properly
                coords = box.xyxy[0].cpu().numpy() if hasattr(box.xyxy, 'cpu') else box.xyxy[0]
                x1, y1, x2, y2 = map(int, coords)
            except (IndexError, AttributeError, TypeError):
                continue
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)
            
            # Reduced minimum size to catch more detections (was 10x10, now 5x5)
            if (x2 - x1) < 5 or (y2 - y1) < 5:
                continue
            
            person_img = Image.fromarray(cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))
            detections.append(person_img)
    
    return detections


def create_result_directory(video_name, username="general"):
    """Create directory for video results with user separation"""
    user_dir = os.path.join(RESULT_DIR, username)
    video_dir = os.path.join(user_dir, video_name)
    os.makedirs(video_dir, exist_ok=True)
    return video_dir


def save_detection_image(frame, target_name, color, gender, video_dir, timestamp, accuracy=0):
    """Save detected person image with accuracy"""
    accuracy_str = f"{accuracy:.1f}%" if accuracy > 0 else "N/A"
    filename = f"Found_{target_name}_{color}_{gender}_{accuracy_str}_{timestamp}.jpg"
    filepath = os.path.join(video_dir, filename)
    cv2.imwrite(filepath, frame)
    return filename


def get_video_properties(video_path):
    """Get video duration and frame rate"""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    cap.release()
    return {
        'total_frames': total_frames,
        'fps': fps,
        'duration': duration
    }
