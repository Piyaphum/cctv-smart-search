# Person Detection System - Architecture

## Project Structure

```
d:\person-reid\
├── app.py                    # Main application (clean, ~300 lines)
├── config.py                 # All configuration constants
├── models.py                 # AI models loading
├── feature_extraction.py      # Extract embeddings, colors, gender
├── target_management.py       # Target profile operations
├── video_processor.py         # Video frame processing
├── search_engine.py          # Matching logic
├── email_service.py          # Email alerts
├── ui_styles.py              # Dark-green theme CSS
├── database.py               # Database operations (existing)
├── auth_config.yaml          # Authentication config (existing)
└── requirements.txt          # Dependencies
```

## File Responsibilities

| File | Purpose |
|------|---------|
| `app.py` | Main Streamlit app, UI layout, orchestration |
| `config.py` | All constants and configuration values |
| `models.py` | Load YOLO, ResNet50, CLIP models |
| `feature_extraction.py` | Embeddings, color analysis, gender detection |
| `target_management.py` | Create & manage target profiles |
| `video_processor.py` | Video frame processing & saving results |
| `search_engine.py` | Person matching logic & similarity |
| `email_service.py` | Send detection alerts via email |
| `ui_styles.py` | Dark green theme styling |
| `database.py` | SQLite operations for profiles |

## Theme Colors

- **Primary Green**: `#1dd1a1` (bright teal)
- **Dark Background**: `#0a0e27` (almost black)
- **Sidebar**: `#0f1419` (dark blue-black)
- **Cards**: `#151b28` (dark gray)
- **Text**: `#e8eef2` (light)
- **Muted**: `#a0aec0` (gray)

## Running the App

```bash
# Install dependencies
pip install -r requirements.txt

# Run streamlit app
python -m streamlit run app.py

# Or use the launcher
python run_app.py
```

## Building Executable

```bash
# Build .exe
python build_exe.py

# Output: dist/PersonReid.exe
```

## Key Features

✅ Dark green modern theme  
✅ Clean modular architecture  
✅ Face recognition with ResNet50  
✅ Object detection with YOLOv8  
✅ CLIP for text-based search  
✅ Saved profile database  
✅ Email alerts on detection  
✅ User authentication  

## Development Notes

- Each module is self-contained and focused on one task
- Models are cached with `@st.cache_resource` for performance
- All constants are in `config.py` for easy modification
- UI theme in `ui_styles.py` for consistent styling
