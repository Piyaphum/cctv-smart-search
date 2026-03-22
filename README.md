# Person Detection and Re-identification System

This project is a comprehensive security application designed to analyze CCTV video footage and identify specific individuals using artificial intelligence. The system allows users to upload "Target Profiles" (images of people of interest) and scans provided video files to locate these individuals based on facial embeddings, clothing colors, and semantic features.

## Origins and Purpose

This system was initially developed as a localized desktop application utilizing a lightweight SQLite database. As the feature set expanded to include real-time multi-tenant administration, public registration portals, and scalable search histories, the architecture was migrated to a Supabase Cloud PostgreSQL database. 

The primary objective of this project is to automate the tedious process of manually scrubbing hours of CCTV footage. By leveraging state-of-the-art vision models (YOLOv8 and OpenAI CLIP), the application dramatically reduces search time and provides instant timestamped logs and automated email alerts.

## Tech Stack

*   **Frontend / UI:** Streamlit (Python). Selected for its rapid prototyping capabilities and seamless native data binding.
*   **Authentication:** `streamlit-authenticator` integrated directly with Supabase via custom handlers.
*   **Database:** Supabase (PostgreSQL). Utilized for `users` management, `search_history` logging, and storing heavy `target_profiles` (JSON format of deep neural network embeddings).
*   **AI & Computer Vision:**
    *   `ultralytics` (YOLOv8): For bounding box detection of humans within video frames.
    *   `transformers` (OpenAI CLIP): For extracting highly-dimensional semantic embeddings from detected human crops to perform similarity matching.
    *   `deepface`: For extracting biological profile classifications.
    *   `opencv-python` (cv2): Standard frame extraction and image filtering.
*   **Scientific Compute:** `numpy`, `pandas`, `scipy` for evaluating cosine similarities between multi-dimensional arrays.

## Installation Guide

1.  **Clone the Repository** and navigate to the root directory.
2.  **Install Dependencies:** Ensure you are using Python 3.9+ and pip.
    ```bash
    pip install -r requirements.txt
    ```
3.  **Configure Environment Variables:** Open `config.py` and populate the necessary credentials:
    *   `SUPABASE_URL`
    *   `SUPABASE_KEY`
    *   `SENDER_EMAIL`
    *   `SENDER_PASSWORD`
4.  **Database Migration (Supabase):** Ensure your Supabase remote SQL instance has the appropriate tables provisioned (`users`, `target_profiles`, `search_history`, `detections`). See `architecture.md` for schema definitions.
5.  **Run the Application:**
    ```bash
    streamlit run app.py
    ```

## Maintenance and Upgrades

*   **Database Backups:** Supabase automatically creates daily snapshots of the database. However, ensure that JSON embeddings in `target_profiles` are periodically pruned if they are no longer relevant to save cloud bandwidth.
*   **Model Caching:** The HuggingFace Transformers library will download the `openai/clip-vit-base-patch32` weights locally the first time it is run. Ensure the host environment has at least 2GB of free storage for model cache files.

## Precautions

1.  **Code Security:** Never commit `config.py` if it contains hardcoded raw API keys or gmail application passwords to public repositories. Always use `.env` files or secure secret managers in production.
2.  **Memory Constraints:** The application processes video frame-by-frame. Setting the `DEFAULT_SNAPSHOT_INTERVAL` too low (e.g., `< 0.5s`) on heavy videos (4K duration) will result in extremely high RAM usage due to array vectorizations.
3.  **Authentication Binding:** The "Forgot Password" 2-step verification requires an active outgoing SMTP server configuration. If Gmail blocks the application (due to Google Account security policy resets), users will not receive the verification code required to regain access. Always verify email settings.
