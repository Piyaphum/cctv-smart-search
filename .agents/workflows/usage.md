---
description: Example workflow for using the Person Re-Identification System
---
This workflow describes the step-by-step process for using the Person Detection System.

1. **User Authentication**
   - Navigate to the application URL.
   - Log in with your username and password.
   - If you don't have an account, use the "Register New User" link.

2. **Setup Targets**
   - Go to the **Search** tab.
   - Under **Target Setup**, click **New Upload**.
   - Upload high-quality images of the person you want to find.
   - Provide a name for the target and click **Save Profile** to store it in the database for future use.
   - Select the target from the **Saved Profiles** list.

3. **Configure Search Parameters** (Optional but Recommended)
   - **Similarity Threshold**: Adjust to control how strictly the system matches faces (default 0.70).
   - **Color Weight**: Adjust how much the system should prioritize clothing color matching (default 0.60).
   - **Scan Interval**: Set the frequency of frame sampling (e.g., 1.0s).

4. **Upload and Process Video**
   - Under **Video Search**, upload one or more CCTV video files (MP4, AVI, etc.).
   - Click **Start Search**.
   - Monitor the progress bar and status messages as the AI scans the footage.

5. **Review Results**
   - Once complete, switch to the **Results** tab.
   - Expand the video section to see the **Search Analytics** dashboard (Metrics and Charts).
   - Browse the detected images. Each image includes:
     - The target name.
     - A match confidence score.
     - A precise timestamp in the video.

6. **Email Alerts**
   - If enabled, the system will send an email report to the specified recipients containing the search summary and match details.
