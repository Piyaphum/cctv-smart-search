"""
Configuration and Constants
"""
import os

# URLs
WEB_APP_URL = "http://localhost:8501"

# Directories
RESULT_DIR = "detected_results"
TEMP_DIR = "D:\\person-reid\\temp_video"
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# Settings
MAX_IMAGES_KEPT = 100
DEFAULT_SIMILARITY_THRESHOLD = 0.70
DEFAULT_COLOR_STRICTNESS = 0.6
DEFAULT_SNAPSHOT_INTERVAL = 1.0

# Email Config
SENDER_EMAIL = "piyaphum1492@gmail.com"
SENDER_PASSWORD = "vhvp varc qflt ryxv"

# Model Paths
YOLO_MODEL = 'yolov8n.pt'
CLIP_MODEL = "openai/clip-vit-base-patch32"