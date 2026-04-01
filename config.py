"""
Configuration and Constants
"""
import os

# URLs
WEB_APP_URL = "https://ku-person-target-searching.streamlit.app/"

# Supabase Auth Connection (Censored for security)
SUPABASE_URL = "https://jghslowximgwaqdocklr.supabase.co/"
SUPABASE_KEY = "sb_publishable_PFpZlGsNzH6rMrwjRIlQkw_3Q7ISRz-"
# Directories
RESULT_DIR = "detected_results"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_DIR = os.path.join(BASE_DIR, "temp_video")
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# Settings
MAX_IMAGES_KEPT = 100
DEFAULT_SIMILARITY_THRESHOLD = 0.70
DEFAULT_COLOR_STRICTNESS = 0.6
DEFAULT_SNAPSHOT_INTERVAL = 1.0

# Email Config (Censored for security)
SENDER_EMAIL = "your_email@gmail.com"
SENDER_PASSWORD = "your_email_app_password"

# Model Paths
YOLO_MODEL = 'yolov8n.pt'
