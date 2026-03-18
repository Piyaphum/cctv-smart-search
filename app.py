"""
Person Detection & Re-identification System
Clean Architecture - Default Streamlit Theme
"""
import streamlit as st
import os
import cv2
from PIL import Image
import glob

# ===== Import Modules =====
from config import RESULT_DIR, TEMP_DIR, DEFAULT_SIMILARITY_THRESHOLD, DEFAULT_COLOR_STRICTNESS, DEFAULT_SNAPSHOT_INTERVAL
from models import get_all_models
from feature_extraction import extract_embedding, get_part_histogram, get_dominant_color_name
from target_management import generate_target_data
from video_processor import save_detection_image, create_result_directory
from email_service import send_email_report
from database import init_db, get_all_target_profiles, delete_target_profile, save_target_profile
from translations import get_text


# ===== Page Config =====
st.set_page_config(
    page_title="Person Detection System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== Authentication =====
import yaml
import streamlit_authenticator as stauth

with open('auth_config.yaml', 'r', encoding='utf-8') as f:
    auth_config = yaml.safe_load(f)

authenticator = stauth.Authenticate(
    auth_config['credentials'],
    auth_config['cookie']['name'],
    auth_config['cookie']['key'],
    auth_config['cookie']['expiry_days']
)

login_result = authenticator.login(location='main')

if not st.session_state.get('authentication_status'):
    if st.session_state.get('authentication_status') is False:
        st.error('Invalid username or password')
    else:
        st.info('Please log in to continue')
    st.stop()

# ===== Setup =====
init_db()
current_user = st.session_state.get('username', 'unknown')
current_name = st.session_state.get('name', 'Unknown')

# Language management
if 'language' not in st.session_state:
    st.session_state.language = 'th'

# Sidebar
with st.sidebar:
    st.markdown("### " + get_text('system', st.session_state.language))
    st.caption(f"{get_text('user', st.session_state.language)}: {current_name}")
    
    # Language selector
    lang_options = {'ไทย': 'th', 'English': 'en'}
    selected_lang = st.selectbox(
        get_text('language', st.session_state.language),
        options=lang_options.keys(),
        index=0 if st.session_state.language == 'th' else 1
    )
    st.session_state.language = lang_options[selected_lang]
    
    authenticator.logout(get_text('logout', st.session_state.language), 'sidebar')
    st.divider()

# Load models once
models = get_all_models()

# ===== Main Header =====
lang = st.session_state.language
st.markdown(f"<h1>{get_text('page_title', lang)}</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='color:#a0aec0;'>{get_text('subtitle', lang)}</p>", unsafe_allow_html=True)

# ===== Tabs =====
tab1, tab2 = st.tabs([get_text('search', lang), get_text('results', lang)])

# ===== TAB 1: SEARCH =====
with tab1:
    col_left, col_right = st.columns([1, 2], gap="large")
    
    # --- LEFT: Target Setup ---
    with col_left:
        st.markdown(f"### {get_text('target_setup', lang)}")
        
        targets_db = []
        t_tab1, t_tab2 = st.tabs([get_text('saved_profiles', lang), get_text('new_upload', lang)])
        
        with t_tab1:
            saved_targets = get_all_target_profiles()
            if saved_targets:
                st.markdown(f"**{get_text('select_profiles', lang)}**")
                for tgt in saved_targets:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        if st.checkbox(f"{tgt['name']}", key=f"sel_{tgt['id']}"):
                            targets_db.append({
                                "name": tgt["name"],
                                "type": "image",
                                "embeddings": tgt["embeddings"],
                                "hists_full": tgt["hists_full"],
                                "hists_top": tgt["hists_top"]
                            })
                    with col2:
                        if st.button(get_text('delete', lang), key=f"del_{tgt['id']}", use_container_width=True):
                            delete_target_profile(tgt['id'])
                            st.rerun()
            else:
                st.info(get_text('no_saved_profiles', lang))
        
        with t_tab2:
            target_files = st.file_uploader(get_text('upload_images', lang), type=['jpg', 'png'], accept_multiple_files=True)
            save_to_db = st.checkbox(get_text('save_to_database', lang), value=True)
            
            if target_files:
                with st.spinner(get_text('processing', lang)):
                    for tfile in target_files:
                        tname = st.text_input(
                            f"{get_text('name_for', lang)} {tfile.name}",
                            value=tfile.name.split('.')[0],
                            key=f"name_{tfile.name}"
                        )
                        
                        tdata = generate_target_data(
                            tfile,
                            models['detector'],
                            models['reid_model'],
                            models['base_transform'],
                            models['aug_transform']
                        )
                        tdata["name"] = tname
                        targets_db.append(tdata)
                        
                        c1, c2 = st.columns([1, 3])
                        c1.image(tdata['image'], use_container_width=True)
                        c2.markdown(f"**{tname}**")
                        
                        if save_to_db and st.button(f"{get_text('save_profile', lang)}: '{tname}'", key=f"save_{tfile.name}"):
                            save_target_profile(
                                name=tname,
                                embeddings=tdata["embeddings"],
                                hists_full=tdata["hists_full"],
                                hists_top=tdata["hists_top"],
                                created_by=current_user
                            )
                            st.success(get_text('saved', lang))
        
        if targets_db:
            st.markdown(f"<span style='color:#1dd1a1;'>✓ {len(targets_db)} {get_text('targets_selected', lang)}</span>", unsafe_allow_html=True)
        
        st.markdown("---")
        enable_email = st.checkbox(get_text('send_alerts', lang), value=True)
        recipient_emails = []
        if enable_email:
            num_recipients = int(st.number_input(get_text('recipients', lang), min_value=1, step=1, value=1))
            for i in range(num_recipients):
                email = st.text_input(f"Email {i+1}", key=f"email_{i}")
                if email:
                    recipient_emails.append(email)
    
    # --- RIGHT: Video Search ---
    with col_right:
        st.markdown(f"### {get_text('video_search', lang)}")
        video_files = st.file_uploader(get_text('upload_videos', lang), type=['mp4', 'avi'], accept_multiple_files=True)
        
        st.markdown(f"**{get_text('parameters', lang)}**")
        c1, c2, c3 = st.columns(3)
        threshold = c1.slider(
            get_text('similarity', lang), 
            0.0, 1.0, 
            DEFAULT_SIMILARITY_THRESHOLD,
            help=get_text('similarity_help', lang)
        )
        shirt_weight = c2.slider(
            get_text('color_weight', lang), 
            0.0, 1.0, 
            DEFAULT_COLOR_STRICTNESS,
            help=get_text('color_weight_help', lang)
        )
        interval = c3.slider(
            get_text('interval', lang), 
            0.5, 5.0, 
            DEFAULT_SNAPSHOT_INTERVAL,
            help=get_text('interval_help', lang)
        )
        
        with st.expander(get_text('how_it_works', lang), expanded=False):
            st.markdown(get_text('how_it_works_text', lang))
        
        if st.button(get_text('start_search', lang), type="primary", use_container_width=True):
            if not video_files or not targets_db:
                st.error(get_text('error_please_upload', lang))
            else:
                st.info("Feature coming soon...")


# ===== TAB 2: RESULTS =====
with tab2:
    st.markdown(f"### {get_text('detection_results', lang)}")
    
    if st.button(get_text('refresh', lang), use_container_width=True):
        st.rerun()
    
    if os.path.exists(RESULT_DIR):
        videos = [f for f in os.listdir(RESULT_DIR) if os.path.isdir(os.path.join(RESULT_DIR, f))]
        
        if not videos:
            st.info(get_text('no_results_yet', lang))
        
        for video in videos:
            with st.expander(f"{video}", expanded=True):
                video_dir = os.path.join(RESULT_DIR, video)
                images = sorted(
                    glob.glob(os.path.join(video_dir, "*.jpg")),
                    key=os.path.getmtime,
                    reverse=True
                )
                
                if images:
                    cols = st.columns(5)
                    for i, img_path in enumerate(images):
                        img = Image.open(img_path)
                        fname = os.path.basename(img_path)
                        parts = fname.split('_')
                        label = parts[1] if len(parts) > 1 else fname
                        
                        cols[i % 5].image(img, use_container_width=True)
                        cols[i % 5].caption(label)
                else:
                    st.caption(get_text('no_detected_results', lang))
    else:
        st.info("Results directory not found")


# ===== Documentation =====
st.markdown("---")
with st.expander(get_text('documentation', lang)):
    st.markdown(f"""
    ### {get_text('quick_start', lang)}
    1. **{get_text('add_targets', lang).split(' - ')[0]}** - {get_text('add_targets', lang).split(' - ')[1]}
    2. **{get_text('upload_video', lang).split(' - ')[0]}** - {get_text('upload_video', lang).split(' - ')[1]}
    3. **{get_text('configure', lang).split(' - ')[0]}** - {get_text('configure', lang).split(' - ')[1]}
    4. **{get_text('search_process', lang).split(' - ')[0]}** - {get_text('search_process', lang).split(' - ')[1]}
    5. **{get_text('review_results', lang).split(' - ')[0]}** - {get_text('review_results', lang).split(' - ')[1]}
    
    ### {get_text('parameters', lang)}
    - **{get_text('similarity', lang)}** - {get_text('similarity_help', lang)}
    - **{get_text('color_weight', lang)}** - {get_text('color_weight_help', lang)}
    - **{get_text('interval', lang)}** - {get_text('interval_help', lang)}
    
    ### {get_text('tips', lang)}
    - {get_text('tip_1', lang)}
    - {get_text('tip_2', lang)}
    - {get_text('tip_3', lang)}
    """)
