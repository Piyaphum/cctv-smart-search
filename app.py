"""
Person Detection & Re-identification System
Clean Architecture - Default Streamlit Theme
"""
import streamlit as st
import os
import shutil
import cv2
from PIL import Image
import glob
import json
import pandas as pd

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
from supabase import create_client, Client
import config

try:
    with open('auth_config.yaml', 'r', encoding='utf-8') as f:
        auth_config = yaml.safe_load(f)
except Exception:
    auth_config = {'cookie': {'name': 'cctv_token_cookie', 'key': 'secret_key_123', 'expiry_days': 30}}

# Fetch users from Supabase Cloud Database dynamically
try:
    supabase: Client = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)
    response = supabase.table('users').select('*').execute()
    
    # Initialize credentials if not exists
    if 'credentials' not in auth_config:
        auth_config['credentials'] = {'usernames': {}}
    
    if response.data:
        # Streamlit-Authenticator uses lowercase usernames internally
        if 'credentials' in auth_config and 'usernames' in auth_config['credentials']:
            auth_config['credentials']['usernames'] = {
                k.lower(): v for k, v in auth_config['credentials']['usernames'].items()
            }
            
        for user in response.data:
            uname = user['username'].strip().lower()
            # Merge: This ensures users in auth_config.yaml are kept if not in DB
            auth_config['credentials']['usernames'][uname] = {
                'email': user['email'],
                'name': user['name'],
                'password': user['password_hash'],
                'role': user.get('role', 'viewer')
            }
except Exception as e:
    st.error(f"Failed to connect to Cloud Database: {e}")
    if 'credentials' not in auth_config:
        auth_config['credentials'] = {'usernames': {}}

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
        
    lang = st.session_state.get('language', 'th')
    
    # --- Register New User ---
    with st.expander(get_text('register_new_user', lang)):
        with st.form("public_register_form"):
            reg_username = st.text_input(get_text('username', lang))
            
            c1, c2 = st.columns(2)
            with c1: reg_fname = st.text_input(get_text('first_name', lang))
            with c2: reg_lname = st.text_input(get_text('last_name', lang))
            
            reg_email = st.text_input(get_text('email', lang))
            reg_password = st.text_input(get_text('password', lang), type="password", help="min 8 chars")
            reg_password_confirm = st.text_input(get_text('confirm_password', lang), type="password")
            
            if st.form_submit_button(get_text('register_button', lang), type="primary"):
                reg_username = reg_username.strip()
                reg_fname = reg_fname.strip()
                reg_lname = reg_lname.strip()
                reg_email = reg_email.strip()
                
                if not reg_username or not reg_fname or not reg_lname or not reg_email or not reg_password:
                    st.error(get_text('fill_all_fields', lang))
                elif len(reg_password) < 8:
                    st.error(get_text('password_length_error', lang))
                elif reg_password != reg_password_confirm:
                    st.error(get_text('passwords_not_match', lang))
                else:
                    try:
                        hashed_pw = stauth.Hasher([reg_password]).generate()[0]
                        reg_name = f"{reg_fname} {reg_lname}"
                        data = {
                            "username": reg_username.lower(),
                            "name": reg_name,
                            "email": reg_email,
                            "password_hash": hashed_pw,
                            "role": "viewer"  # Default public role is always viewer
                        }
                        if supabase:
                            supabase.table('users').insert(data).execute()
                            st.success(get_text('registration_success', lang))
                            # Add a small delay and rerun to ensure DB state is updated and fetched
                            import time
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("Cloud database is unavailable.")
                    except Exception as e:
                        if "duplicate" in str(e).lower() or "conflict" in str(e).lower():
                            st.error(get_text('user_creation_error', lang))
                        else:
                            st.error(f"Registration Error: {e}")

    # Initialize reset state variables
    if 'reset_step' not in st.session_state:
        st.session_state.reset_step = 1
        st.session_state.reset_target_user = None
        st.session_state.reset_target_email = None
        st.session_state.reset_code = None

    # --- Forgot Password ---
    with st.expander(get_text('forgot_password', lang)):
        if st.session_state.reset_step == 1:
            with st.form("forgot_password_step1"):
                st.markdown("**1. ยืนยันตัวตน (Identify Account)**")
                identifier = st.text_input("Username or Email (ชื่อผู้ใช้ หรือ อีเมล)")
                
                if st.form_submit_button("ส่งรหัสยืนยัน (Send Code)", type="primary"):
                    if not identifier.strip():
                        st.warning("กรุณากรอก Username หรือ Email")
                    else:
                        target_uname = None
                        target_email = None
                        
                        # Find user in Supabase
                        if supabase:
                            # Username is case-insensitive for lookup
                            res_u = supabase.table('users').select('*').ilike('username', identifier.strip()).execute()
                            if res_u.data:
                                target_uname = res_u.data[0]['username']
                                target_email = res_u.data[0]['email']
                            else:
                                res_e = supabase.table('users').select('*').ilike('email', identifier.strip()).execute()
                                if res_e.data:
                                    target_uname = res_e.data[0]['username']
                                    target_email = res_e.data[0]['email']
                        
                        # Fallback
                        if not target_uname:
                            for un, dt in auth_config['credentials']['usernames'].items():
                                if un.lower() == identifier.strip().lower() or dt.get('email', '').lower() == identifier.strip().lower():
                                    target_uname = un
                                    target_email = dt.get('email')
                                    break
                                    
                        if target_uname and target_email:
                            import random
                            vcode = f"{random.randint(100000, 999999)}"
                            st.session_state.reset_target_user = target_uname
                            st.session_state.reset_target_email = target_email
                            st.session_state.reset_code = vcode
                            
                            from email_service import send_verification_code_email
                            success, msg = send_verification_code_email(target_email, target_uname, vcode)
                            if success:
                                st.session_state.reset_step = 2
                                st.rerun()
                            else:
                                st.error(f"Failed to send email: {msg}")
                        else:
                            st.error(get_text('username_not_found', lang))
                            
        elif st.session_state.reset_step == 2:
            st.info(f"ระบบได้ส่งรหัสยืนยัน 6 หลักไปที่อีเมล: **{st.session_state.reset_target_email}** แล้ว")
            with st.form("forgot_password_step2"):
                st.markdown("**2. ตั้งรหัสผ่านใหม่ (Reset Password)**")
                entered_code = st.text_input("รหัสยืนยัน 6 หลัก (6-digit Code)")
                new_pass = st.text_input("รหัสผ่านใหม่ (New Password)", type="password", help="อย่างน้อย 8 ตัวอักษร")
                new_pass_confirm = st.text_input("ยืนยันรหัสผ่านใหม่ (Confirm Password)", type="password")
                
                c1, c2 = st.columns(2)
                with c1:
                    submit_reset = st.form_submit_button("เปลี่ยนรหัสผ่าน (Confirm Reset)", type="primary")
                with c2:
                    cancel_reset = st.form_submit_button("ยกเลิก (Cancel)")
                
                if submit_reset:
                    if entered_code != st.session_state.reset_code:
                        st.error("รหัสยืนยันไม่ถูกต้อง (Invalid Code)")
                    elif len(new_pass) < 8:
                        st.error(get_text('password_length_error', lang))
                    elif new_pass != new_pass_confirm:
                        st.error(get_text('passwords_not_match', lang))
                    else:
                        new_hashed_pw = stauth.Hasher([new_pass]).generate()[0]
                        uname = st.session_state.reset_target_user
                        
                        try:
                            if supabase:
                                supabase.table('users').update({'password_hash': new_hashed_pw}).eq('username', uname).execute()
                                
                            if uname in auth_config['credentials']['usernames']:
                                auth_config['credentials']['usernames'][uname]['password'] = new_hashed_pw
                                with open('auth_config.yaml', 'w', encoding='utf-8') as f:
                                    import yaml
                                    yaml.dump(auth_config, f, default_flow_style=False, sort_keys=False)
                            
                            st.success("เปลี่ยนรหัสผ่านสำเร็จแล้ว! กรุณาล็อกอินด้วยรหัสผ่านใหม่")
                            st.session_state.reset_step = 1
                            st.session_state.reset_target_user = None
                            st.session_state.reset_code = None
                        except Exception as e:
                            st.error(f"Error resolving password change: {e}")
                            
                if cancel_reset:
                    st.session_state.reset_step = 1
                    st.session_state.reset_target_user = None
                    st.session_state.reset_code = None
                    st.rerun()
                    
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

    # --- System Config (Moved from Admin Panel) ---
    with st.expander("ตั้งค่า Email"):
        st.markdown("**ตั้งค่าอีเมลส่งแจ้งเตือน**")
        st.caption("ระบบใช้อีเมลนี้ส่งรหัส OTP และแจ้งเตือน")
        
        with st.popover("📖 คู่มือขอรหัสผ่านแอป Gmail"):
            st.markdown("""
            **วิธีสร้างรหัส App Password:**
            1. ไปที่บัญชี Google > **ความปลอดภัย (Security)**
            2. เปิดใช้งาน **การยืนยันแบบ 2 ขั้นตอน (2-Step Verification)**
            3. ค้นหาเมนู **"App Passwords"**
            4. กรอกชื่อแอป แล้วกดสร้าง
            5. นำรหัส 16 ตัวมากรอก
            """)
            
        try:
            import yaml
            if os.path.exists('user_settings.yaml'):
                with open('user_settings.yaml', 'r', encoding='utf-8') as f:
                    all_user_settings = yaml.safe_load(f) or {}
            else:
                all_user_settings = {}
        except Exception:
            all_user_settings = {}
            
        user_config = all_user_settings.get(current_user, {})
        curr_email = user_config.get("SENDER_EMAIL", "")
        curr_pass = user_config.get("SENDER_PASSWORD", "")
        
        # Load global settings as fallback if user has none
        if not curr_email:
            try:
                with open('settings.yaml', 'r', encoding='utf-8') as f:
                    sys_settings = yaml.safe_load(f) or {}
                curr_email = sys_settings.get("SENDER_EMAIL", "")
                curr_pass = sys_settings.get("SENDER_PASSWORD", "")
            except:
                pass

        # Use container instead of form to avoid "Press Enter to submit form" hint
        with st.container():
            sys_email = st.text_input("Sender Email", value=curr_email, key=f"email_input_{current_user}")
            sys_pass = st.text_input("App Password", value=curr_pass, type="password", key=f"pass_input_{current_user}")
            
            if st.button("Save Settings", type="primary", key=f"save_btn_{current_user}"):
                if current_user not in all_user_settings:
                    all_user_settings[current_user] = {}
                
                all_user_settings[current_user]["SENDER_EMAIL"] = sys_email.strip()
                if sys_pass.strip():  
                    all_user_settings[current_user]["SENDER_PASSWORD"] = sys_pass.strip().replace(" ", "")
                
                try:
                    import yaml
                    with open('user_settings.yaml', 'w', encoding='utf-8') as f:
                        yaml.dump(all_user_settings, f, default_flow_style=False)
                    st.success("บันทึกสำเร็จ!")
                    # Briefly wait or rerun to refresh view if needed
                    # st.rerun() 
                except Exception as e:
                    st.error(f"Error: {e}")

    st.divider()
    
    # Personal cache management (Admin only)
    user_role = auth_config['credentials']['usernames'].get(current_user, {}).get('role', 'viewer')
    if user_role == 'admin':
        st.markdown("### " + ("Clear Personal Cache" if st.session_state.language == 'en' else "ล้างแคชส่วนตัว"))
        
        def get_user_cache_size():
            """Get cache size for current user"""
            user_cache_dir = os.path.join(RESULT_DIR, current_user)
            if not os.path.exists(user_cache_dir):
                return 0
            total = 0
            try:
                for root, dirs, files in os.walk(user_cache_dir):
                    for file in files:
                        total += os.path.getsize(os.path.join(root, file))
            except:
                pass
            return total / (1024 * 1024)  # Convert to MB
        
        cache_size = get_user_cache_size()
        st.info(f"{cache_size:.2f} MB" if st.session_state.language == 'en' else f"{cache_size:.2f} MB")
        
        if st.button(
            "Clear My Cache" if st.session_state.language == 'en' else "ล้างแคชของฉัน",
            type="secondary",
            use_container_width=True
        ):
            user_cache_dir = os.path.join(RESULT_DIR, current_user)
            if os.path.exists(user_cache_dir):
                try:
                    import shutil
                    shutil.rmtree(user_cache_dir)
                    os.makedirs(user_cache_dir)
                    st.success("Cache cleared!" if st.session_state.language == 'en' else "ล้างแคชเรียบร้อยแล้ว!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")
            else:
                st.info("No cache to clear" if st.session_state.language == 'en' else "ไม่มีแคชให้ล้าง")
        
        st.divider()

# Load models once
models = get_all_models()

# ===== Main Header =====
lang = st.session_state.language
st.markdown(f"<h1>{get_text('page_title', lang)}</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='color:#a0aec0;'>{get_text('subtitle', lang)}</p>", unsafe_allow_html=True)

# ===== Get User Role =====
current_role = auth_config['credentials']['usernames'].get(current_user, {}).get('role', 'viewer')

# ===== Tabs =====
view_param = st.query_params.get("view", "")

if view_param == "results":
    tab2, tab1 = st.tabs([get_text('results', lang), get_text('search', lang)])
else:
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
            st.markdown(f"<span style='color:#1dd1a1;'>{len(targets_db)} {get_text('targets_selected', lang)}</span>", unsafe_allow_html=True)
        
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
                from search_engine import batch_match_targets
                from feature_extraction import extract_embedding, get_part_histogram, get_dominant_color_name
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                results_container = st.empty()
                
                total_detections = 0
                total_matches = 0
                email_summary = {}  # Track matches for email
                
                for video_file in video_files:
                    video_name = video_file.name.split('.')[0]
                    video_bytes = video_file.read()
                    
                    # Save temp video
                    temp_video_path = os.path.join(TEMP_DIR, f"{video_name}_temp.mp4")
                    with open(temp_video_path, 'wb') as f:
                        f.write(video_bytes)
                    
                    # Create result directory for this user
                    video_result_dir = create_result_directory(video_name, current_user)
                    email_summary[video_name] = {}
                    
                    # Track search metadata for summary visualization
                    search_metadata = {
                        "video_name": video_name,
                        "total_detections": 0,
                        "matches": [],
                        "target_counts": {}
                    }
                    
                    try:
                        cap = cv2.VideoCapture(temp_video_path)
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        frame_interval = max(1, int(fps * interval))
                        
                        frame_count = 0
                        processed_frames = 0
                        
                        while cap.isOpened():
                            ret, frame = cap.read()
                            if not ret:
                                break
                            
                            # Process every Nth frame based on interval
                            if frame_count % frame_interval == 0:
                                # Detect persons in frame with lower confidence threshold
                                results = models['detector'](frame, classes=0, conf=0.3, verbose=False)
                                
                                # If no detections with 0.3 confidence, try with lower threshold
                                if not results or len(results) == 0:
                                    results = models['detector'](frame, classes=0, conf=0.15, verbose=False)
                                
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
                                        x1, y1 = max(0, x1), max(0, y1)
                                        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                                        
                                        # Reduced minimum size to catch more detections (was 10x10, now 5x5)
                                        if (x2 - x1) >= 5 and (y2 - y1) >= 5:
                                            person_frame = frame[y1:y2, x1:x2]
                                            person_img_pil = Image.fromarray(cv2.cvtColor(person_frame, cv2.COLOR_BGR2RGB))
                                            
                                            total_detections += 1
                                            search_metadata["total_detections"] += 1
                                            
                                            # Extract features
                                            embedding = extract_embedding(
                                                person_img_pil,
                                                models['reid_model'],
                                                models['base_transform']
                                            )
                                            
                                            hist = get_part_histogram(person_img_pil)
                                            
                                            # Try to detect gender
                                            gender = "Unknown"
                                            
                                            # Match against targets
                                            matches = batch_match_targets(
                                                [embedding],
                                                hist,
                                                targets_db,
                                                threshold=threshold,
                                                shirt_strictness=shirt_weight
                                            )
                                            
                                            if matches:
                                                total_matches += len(matches)
                                                for match in matches:
                                                    color_name = get_dominant_color_name(person_img_pil)
                                                    
                                                    # Calculate timestamp: mm:ss
                                                    seconds = frame_count / fps
                                                    mins = int(seconds // 60)
                                                    secs = int(seconds % 60)
                                                    time_str = f"{mins:02d}m{secs:02d}s"
                                                    full_timestamp = f"{time_str}_{frame_count:06d}"
                                                    
                                                    accuracy_percent = match['similarity'] * 100
                                                    save_detection_image(
                                                        person_frame,
                                                        match['target_name'],
                                                        color_name,
                                                        gender,
                                                        video_result_dir,
                                                        full_timestamp,
                                                        accuracy_percent
                                                    )
                                                    
                                                    # Track for email
                                                    target_name = match['target_name']
                                                    if target_name not in email_summary[video_name]:
                                                        email_summary[video_name][target_name] = []
                                                        email_summary[video_name][target_name].append({
                                                            "color": color_name,
                                                            "gender": gender,
                                                            "accuracy": accuracy_percent
                                                        })
                                                        
                                                        # Track metadata for visualization
                                                        search_metadata["matches"].append({
                                                            "target": target_name,
                                                            "score": match['similarity'],
                                                            "timestamp": seconds,
                                                            "color": color_name,
                                                            "gender": gender
                                                        })
                                                        search_metadata["target_counts"][target_name] = search_metadata["target_counts"].get(target_name, 0) + 1
                            
                            frame_count += 1
                            processed_frames += 1
                            progress = min(processed_frames / (total_frames // frame_interval), 1.0)
                            progress_bar.progress(progress)
                            status_text.text(f"Video: {video_name} | Detections: {total_detections} | Matches: {total_matches}")
                        
                        cap.release()
                    
                    except Exception as e:
                        st.error(f"Error processing video {video_name}: {str(e)}")
                    
                    finally:
                        # Save summary metadata file for Results visualization
                        try:
                            summary_path = os.path.join(video_result_dir, "summary.json")
                            with open(summary_path, 'w', encoding='utf-8') as f:
                                json.dump(search_metadata, f, indent=4, ensure_ascii=False)
                        except:
                            pass
                            
                        # Clean up temp file
                        if os.path.exists(temp_video_path):
                            os.remove(temp_video_path)
                        
                        # === LOG SEARCH HISTORY ===
                        try:
                            from database import log_search
                            for t in targets_db:
                                t_name = t['name']
                                t_found = len(email_summary.get(video_name, {}).get(t_name, []))
                                log_search(
                                    username=current_user,
                                    video_name=video_name,
                                    target_name=t_name,
                                    total_found=t_found
                                )
                        except Exception as db_err:
                            st.warning(f"Could not save search history: {str(db_err)}")
                
                # Final results
                progress_bar.progress(1.0)
                
                # Send email report if enabled
                if enable_email and recipient_emails and total_matches > 0:
                    try:
                        success, msg = send_email_report(email_summary, recipient_emails, username=current_user)
                        if success:
                            st.info(f"Email sent to {', '.join(recipient_emails)}")
                        else:
                            st.warning(f"Email not sent: {msg}")
                    except Exception as e:
                        st.warning(f"Could not send email: {str(e)}")
                
                if total_matches > 0:
                    st.success(f"Search Complete! Found {total_matches} matches in {total_detections} detections")
                else:
                    st.info(f"No matches found in {total_detections} detections")


# ===== TAB 2: RESULTS =====
with tab2:
    st.markdown(f"### {get_text('detection_results', lang)}")
    
    if st.button(get_text('refresh', lang), use_container_width=True):
        st.rerun()
    
    user_result_dir = os.path.join(RESULT_DIR, current_user)
    
    if os.path.exists(user_result_dir):
        videos = [f for f in os.listdir(user_result_dir) if os.path.isdir(os.path.join(user_result_dir, f))]
        
        if not videos:
            st.info(get_text('no_results_yet', lang))
        
        for video in videos:
            with st.expander(f"Video: {video}", expanded=True):
                video_dir = os.path.join(user_result_dir, video)
                
                # --- Dashbord Summary ---
                summary_path = os.path.join(video_dir, "summary.json")
                if os.path.exists(summary_path):
                    with open(summary_path, 'r', encoding='utf-8') as f:
                        m_data = json.load(f)
                    
                    st.markdown("#### " + ("Search Analytics" if lang == 'en' else "ข้อมูลสรุปการค้นหา"))
                    c1, c2, c3 = st.columns(3)
                    c1.metric(get_text('total_found', lang), len(m_data.get("matches", [])))
                    c2.metric("Total People Found", m_data.get("total_detections", 0))
                    precision = (len(m_data.get("matches", [])) / m_data.get("total_detections", 1) * 100)
                    c3.metric("Discovery Rate", f"{precision:.1f}%")
                    
                    if m_data.get("matches"):
                        df = pd.DataFrame(m_data["matches"])
                        
                        col_chart1, col_chart2 = st.columns(2)
                        with col_chart1:
                            st.caption("Matches per Target" if lang == 'en' else "จำนวนที่พบแยกตามเป้าหมาย")
                            target_counts = df['target'].value_counts()
                            st.bar_chart(target_counts)
                        
                        with col_chart2:
                            st.caption("Detection Timeline (seconds)" if lang == 'en' else "ช่วงเหตุการณ์ที่พบ (วินาที)")
                            # Format for timeline: count occurrences in 5s buckets
                            df['time_bucket'] = (df['timestamp'] // 5) * 5
                            timeline_data = df.groupby('time_bucket').size()
                            st.area_chart(timeline_data)
                    st.divider()
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
                        # Parse filename: Found_targetname_color_gender_accuracy%_time_frame.jpg
                        parts = fname.replace('.jpg', '').split('_')
                        target_name = parts[1] if len(parts) > 1 else "Unknown"
                        accuracy_str = parts[4] if len(parts) > 4 else "N/A"  # accuracy% is at index 4
                        
                        # New timestamp parsing (parts[5] is time_str e.g. 01m38s)
                        time_info = ""
                        if len(parts) > 5 and 'm' in parts[5] and 's' in parts[5]:
                            time_str_val = parts[5].replace('m', ':').replace('s', '')
                            time_info = f" | {time_str_val}"
                        
                        # Format display: target name + accuracy + time
                        display_label = f"{target_name}\n {accuracy_str}{time_info}"
                        
                        cols[i % 5].image(img, use_container_width=True)
                        cols[i % 5].caption(display_label)
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