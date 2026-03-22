# System Architecture

This document details the internal module structure, database schemas, and the operational data flow inside the Person Detection System.

## Module Structure

*   `app.py`: The root orchestrator. Renders the main Streamlit loops, hosts the public registration system, processes login authentications, and acts as the entry point for Video Uploads and Settings.
*   `config.py`: The central registry for API endpoints, Supabase credentials, Email configs, and global search thresholds.
*   `database.py`: The database adapter. Translates Python models, `numpy` matrices, and dictionaries dynamically into JSON-compatible arrays transmitted via the Supabase Client API.
*   `email_service.py`: Dispatches SMTP commands to send verification OTP emails for forgotten passwords and comprehensive matched-target summaries.
*   `feature_extraction.py`: Contains AI evaluation layers using YOLOv8, OpenAI CLIP, and DeepFace. Handles bounding box cropping and semantic mathematical transformations.
*   `models.py`: Initializes the AI Neural Networks globally into memory cache on boot to prevent repetitive load times during inference.
*   `video_processor.py`: Conducts the actual chronological sweeping logic of the video file. Iterates mathematically through timestamps, offloading frame data to `feature_extraction.py`, and recording hits back to `database.py`.
*   `pages/admin.py`: Isolated protected portal strictly for users with the `admin` role. Modifies the primary Cloud configuration of system users directly.

## Principles of Operation

1.  **Profile Generation:** The admin uploads an image of a Target. `feature_extraction.py` parses the photo, translating aesthetic properties (colors, dimensions) into deep mathematical vectors (embeddings) and saving them permanently to Supabase via `database.py`.
2.  **Inference Traversal:** When a video is submitted, `video_processor.py` slices the video into interval frames. YOLO identifies humans in each image.
3.  **Semantic Search:** Each cropped human from YOLO undergoes the same transformation by OpenAI CLIP. The system calculates the "Cosine Similarity" against the saved `target_profiles`. If the mathematical distance exceeds the User Threshold, a "Match" is registered.
4.  **Logging and Alerting:** Matches are logged back into the Supabase Cloud. `email_service` aggregates the results into an HTML report format and sends automated dispatches to security teams.

## Database Schema (Supabase PostgreSQL)

The backend relational schema prioritizes flexible logging with UUID structures.

### 1. `users` Table
Stores authentication and credential definitions for all platform tenants.
*   `username` (TEXT, PRIMARY KEY): The login identifier.
*   `name` (TEXT): The full name of the user.
*   `email` (TEXT): Critical for OTP password resets.
*   `password_hash` (TEXT): Encrypted bcrypt strings.
*   `role` (TEXT): Either `viewer` or `admin`.

### 2. `target_profiles` Table
Stores serialized neural network definitions of target subjects.
*   `id` (UUID, PRIMARY KEY): Generated default UUID v4.
*   `name` (TEXT): Human-readable name of the target.
*   `type` (TEXT): Profile type specification.
*   `embeddings` (JSONB): Numpy arrays transformed to JSON list-float datasets.
*   `hists_full` / `hists_top` (JSONB): Histographical array definitions for fallback color filtering.
*   `created_by` (TEXT): Associates the creator.

### 3. `search_history` Table
Logs broad metadata indicating an ongoing or completed sweep by a user.
*   `id` (UUID, PRIMARY KEY): Default UUID v4.
*   `username` (TEXT): Identifying which operator executed the search.
*   `video_name` / `target_name` (TEXT): Context descriptions.
*   `total_found` (INTEGER): Incrementing aggregate counter.

### 4. `detections` Table
Chronological children of `search_history` detailing exactly when and how accurately a human was matched.
*   `id` (BIGINT, PRIMARY KEY): Sequential identification point.
*   `search_id` (UUID): Foreign key linking backwards to `search_history` with cascading deletion rules.
*   `score` (REAL): The cosine similarity float margin (0.0 to 1.0)
*   `timestamp_s` (REAL): The sequential seconds marker inside the uploaded video referencing the match.
