"""
UI Styling - Dark Green Theme
"""

DARK_GREEN_THEME = """
    <style>
        /* Color Variables */
        :root {
            --primary-green: #1dd1a1;
            --dark-bg: #0a0e27;
            --sidebar-bg: #0f1419;
            --card-bg: #151b28;
            --text-light: #e8eef2;
            --text-muted: #a0aec0;
            --border-color: #2d3748;
        }
        
        /* Main background */
        .stApp {
            background-color: var(--dark-bg);
            color: var(--text-light);
        }
        
        .main {
            background-color: var(--dark-bg);
        }
        
        /* Headers */
        h1, h2, h3, h4 {
            color: var(--primary-green) !important;
            font-weight: 700 !important;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
        }
        
        /* Sidebar */
        [data-testid="stSidebar"] {
            background-color: var(--sidebar-bg) !important;
            border-right: 1px solid var(--border-color) !important;
        }
        
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: var(--primary-green) !important;
        }
        
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] div,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] label {
            color: var(--text-light) !important;
        }
        
        /* Buttons */
        .stButton>button {
            border-radius: 8px !important;
            transition: all 0.2s ease !important;
            font-weight: 600 !important;
            border: none !important;
            padding: 10px 24px !important;
            font-size: 14px !important;
        }
        
        .stButton>button[kind="primary"] {
            background-color: var(--primary-green) !important;
            color: #000 !important;
        }
        
        .stButton>button[kind="primary"]:hover {
            background-color: #00d289 !important;
            box-shadow: 0 4px 12px rgba(29, 209, 161, 0.4) !important;
        }
        
        .stButton>button[kind="secondary"] {
            background-color: var(--card-bg) !important;
            color: var(--primary-green) !important;
            border: 1.5px solid var(--primary-green) !important;
        }
        
        .stButton>button[kind="secondary"]:hover {
            background-color: rgba(29, 209, 161, 0.1) !important;
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px !important;
            background-color: transparent !important;
            border-bottom: 1px solid var(--border-color) !important;
        }
        
        .stTabs [data-baseweb="tab"] {
            background-color: transparent !important;
            border-radius: 6px 6px 0 0 !important;
            padding: 12px 18px !important;
            border: none !important;
            border-bottom: 2px solid transparent !important;
            color: var(--text-muted) !important;
            font-weight: 600 !important;
            font-size: 14px !important;
            transition: all 0.2s ease !important;
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            color: var(--primary-green) !important;
            border-bottom-color: var(--primary-green) !important;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: transparent !important;
            color: var(--primary-green) !important;
            border-bottom: 2px solid var(--primary-green) !important;
        }
        
        /* Tab Panel */
        .stTabs [data-baseweb="tab-panel"] {
            border: none !important;
            border-top: 1px solid var(--border-color) !important;
            padding: 20px 0 !important;
        }
        
        /* Input Fields */
        .stTextInput > div > div > input,
        .stNumberInput > div > div > input,
        .stSelectbox > div > div,
        .stFileUploader > div > div {
            border-radius: 6px !important;
            border: 1.5px solid var(--border-color) !important;
            background-color: var(--card-bg) !important;
            color: var(--text-light) !important;
            padding: 10px 12px !important;
        }
        
        .stTextInput > div > div > input:focus,
        .stNumberInput > div > div > input:focus {
            border-color: var(--primary-green) !important;
            box-shadow: 0 0 0 3px rgba(29, 209, 161, 0.2) !important;
        }
        
        /* Sliders */
        .stSlider > div > div > div > div {
            background-color: var(--primary-green) !important;
        }
        
        /* Checkbox & Radio */
        .stCheckbox > label > div {
            background-color: var(--card-bg) !important;
        }
        
        /* Messages */
        .stSuccess {
            background-color: rgba(29, 209, 161, 0.1) !important;
            border: 1px solid rgba(29, 209, 161, 0.3) !important;
            border-radius: 8px !important;
        }
        
        .stError {
            background-color: rgba(255, 71, 87, 0.1) !important;
            border: 1px solid rgba(255, 71, 87, 0.3) !important;
            border-radius: 8px !important;
        }
        
        .stWarning {
            background-color: rgba(255, 193, 7, 0.1) !important;
            border: 1px solid rgba(255, 193, 7, 0.3) !important;
            border-radius: 8px !important;
        }
        
        .stInfo {
            background-color: rgba(66, 165, 245, 0.1) !important;
            border: 1px solid rgba(66, 165, 245, 0.3) !important;
            border-radius: 8px !important;
        }
        
        /* Caption & Labels */
        .stCaption {
            color: var(--text-muted) !important;
            font-size: 13px !important;
        }
        
        /* Expander */
        .streamlit-expanderHeader {
            background-color: var(--card-bg) !important;
            border: 1px solid var(--border-color) !important;
        }
    </style>
    """

def apply_theme():
    """Apply dark green theme"""
    import streamlit as st
    st.markdown(DARK_GREEN_THEME, unsafe_allow_html=True)
