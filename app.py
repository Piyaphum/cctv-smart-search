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
        
    lang = st.session_state.get('language', 'th')
    with st.expander(get_text('forgot_password', lang)):
        with st.form("forgot_password_form"):
            forgot_username = st.text_input(get_text('username', lang))
            if st.form_submit_button(get_text('reset_password', lang)):
                if forgot_username in auth_config['credentials']['usernames']:
                    user_email = auth_config['credentials']['usernames'][forgot_username]['email']
                    import random, string
                    new_random_pass = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
                    
                    # Update config
                    auth_config['credentials']['usernames'][forgot_username]['password'] = stauth.Hasher([new_random_pass]).generate()[0]
                    with open('auth_config.yaml', 'w', encoding='utf-8') as f:
                        yaml.dump(auth_config, f, default_flow_style=False, sort_keys=False)
                    
                    # Send email
                    from email_service import send_password_reset_email
                    success, msg = send_password_reset_email(user_email, forgot_username, new_random_pass)
                    if success:
                        st.success(get_text('reset_success', lang))
                    else:
                        st.error(f"{get_text('reset_fail', lang)}: {msg}")
                else:
                    st.error(get_text('username_not_found', lang))
                    
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
    
    # Personal cache management
    st.markdown("### 🧹 " + ("Clear Personal Cache" if st.session_state.language == 'en' else "ล้างแคชส่วนตัว"))
    
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
    st.info(f"📊 {cache_size:.2f} MB" if st.session_state.language == 'en' else f"📊 {cache_size:.2f} MB")
    
    if st.button(
        "🗑️ Clear My Cache" if st.session_state.language == 'en' else "🗑️ ล้างแคชของฉัน",
        type="secondary",
        use_container_width=True
    ):
        user_cache_dir = os.path.join(RESULT_DIR, current_user)
        if os.path.exists(user_cache_dir):
            try:
                import shutil
                shutil.rmtree(user_cache_dir)
                os.makedirs(user_cache_dir)
                st.success("✅ Cache cleared!" if st.session_state.language == 'en' else "✅ ล้างแคชเรียบร้อยแล้ว!")
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
if current_role == 'admin':
    tab1, tab2, tab3 = st.tabs([get_text('search', lang), get_text('results', lang), get_text('admin_dashboard', lang)])
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
                                # Detect persons in frame
                                results = models['detector'](frame, classes=0, verbose=False)
                                
                                for r in results:
                                    boxes = r.boxes
                                    for box in boxes:
                                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                                        x1, y1 = max(0, x1), max(0, y1)
                                        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                                        
                                        if (x2 - x1) >= 10 and (y2 - y1) >= 10:
                                            person_frame = frame[y1:y2, x1:x2]
                                            person_img_pil = Image.fromarray(cv2.cvtColor(person_frame, cv2.COLOR_BGR2RGB))
                                            
                                            total_detections += 1
                                            
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
                                                    timestamp = f"{frame_count:06d}"
                                                    accuracy_percent = match['similarity'] * 100
                                                    save_detection_image(
                                                        person_frame,
                                                        match['target_name'],
                                                        color_name,
                                                        gender,
                                                        video_result_dir,
                                                        timestamp,
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
                            
                            frame_count += 1
                            processed_frames += 1
                            progress = min(processed_frames / (total_frames // frame_interval), 1.0)
                            progress_bar.progress(progress)
                            status_text.text(f"🎉 Video: {video_name} | Detections: {total_detections} | Matches: {total_matches}")
                        
                        cap.release()
                    
                    except Exception as e:
                        st.error(f"Error processing video {video_name}: {str(e)}")
                    
                    finally:
                        # Clean up temp file
                        if os.path.exists(temp_video_path):
                            os.remove(temp_video_path)
                
                # Final results
                progress_bar.progress(1.0)
                
                # Send email report if enabled
                if enable_email and recipient_emails and total_matches > 0:
                    try:
                        success, msg = send_email_report(email_summary, recipient_emails)
                        if success:
                            st.info(f"📧 Email sent to {', '.join(recipient_emails)}")
                        else:
                            st.warning(f"Email not sent: {msg}")
                    except Exception as e:
                        st.warning(f"Could not send email: {str(e)}")
                
                if total_matches > 0:
                    st.balloons()
                    st.success(f"✅ Search Complete! Found {total_matches} matches in {total_detections} detections")
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
            with st.expander(f"{video}", expanded=True):
                video_dir = os.path.join(user_result_dir, video)
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
                        # Parse filename: Found_targetname_color_gender_accuracy%_timestamp.jpg
                        parts = fname.replace('.jpg', '').split('_')
                        target_name = parts[1] if len(parts) > 1 else "Unknown"
                        accuracy_str = parts[4] if len(parts) > 4 else "N/A"  # accuracy% is at index 4
                        
                        # Format display: target name + accuracy
                        display_label = f"{target_name}\n📊 {accuracy_str}"
                        
                        cols[i % 5].image(img, use_container_width=True)
                        cols[i % 5].caption(display_label)
                else:
                    st.caption(get_text('no_detected_results', lang))
    else:
        st.info("Results directory not found")


# ===== TAB 3: ADMIN DASHBOARD =====
if current_role == 'admin':
    with tab3:
        st.markdown(f"### {get_text('admin_dashboard', lang)}")
        
        t_add, t_manage = st.tabs([get_text('add_new_user', lang), get_text('manage_users', lang)])
        
        with t_add:
            with st.form("new_user_form"):
                st.markdown(f"**{get_text('add_new_user', lang)}**")
                new_username = st.text_input(get_text('username', lang))
                
                col1, col2 = st.columns(2)
                with col1:
                    new_fname = st.text_input(get_text('first_name', lang))
                with col2:
                    new_lname = st.text_input(get_text('last_name', lang))
                new_name = f"{new_fname.strip()} {new_lname.strip()}".strip()
                
                new_email = st.text_input(get_text('email', lang))
                new_password = st.text_input(get_text('password', lang), type="password")
                
                role_options = {
                    get_text('role_viewer', lang): 'viewer',
                    get_text('role_admin', lang): 'admin'
                }
                new_role_label = st.selectbox(get_text('role', lang), options=list(role_options.keys()))
                
                submit_button = st.form_submit_button(get_text('create_user', lang))
                
                if submit_button:
                    if not new_username or not new_fname.strip() or not new_lname.strip() or not new_email or not new_password:
                        st.error(get_text('fill_all_fields', lang))
                    elif len(new_password) < 8:
                        st.error(get_text('password_length_error', lang))
                    elif new_username in auth_config['credentials']['usernames']:
                        st.error(get_text('user_creation_error', lang))
                    else:
                        try:
                            hashed_password = stauth.Hasher([new_password]).generate()[0]
                            auth_config['credentials']['usernames'][new_username] = {
                                'email': new_email,
                                'name': new_name,
                                'password': hashed_password,
                                'role': role_options[new_role_label]
                            }
                            
                            with open('auth_config.yaml', 'w', encoding='utf-8') as f:
                                yaml.dump(auth_config, f, default_flow_style=False, sort_keys=False)
                                
                            st.success(get_text('user_created_success', lang))
                        except Exception as e:
                            st.error(f"Error saving user: {e}")

        with t_manage:
            st.markdown(f"**{get_text('manage_users', lang)}**")
            user_list = list(auth_config['credentials']['usernames'].keys())
            
            for uname in user_list:
                udetails = auth_config['credentials']['usernames'][uname]
                c1, c2, c3, c4 = st.columns([2, 3, 2, 2])
                c1.write(f"**{uname}**")
                c2.write(udetails.get('email', '-'))
                c3.write(udetails.get('role', 'viewer'))
                
                if uname == current_user:
                    c4.caption(f"({get_text('current_user_label', lang)})")
                else:
                    if c4.button(get_text('delete_user', lang), key=f"del_{uname}"):
                        del auth_config['credentials']['usernames'][uname]
                        with open('auth_config.yaml', 'w', encoding='utf-8') as f:
                            yaml.dump(auth_config, f, default_flow_style=False, sort_keys=False)
                        st.rerun()
            
            st.markdown("---")
            st.markdown(f"**{get_text('edit_user', lang)}**")
            user_to_edit = st.selectbox(get_text('select_user_to_edit', lang), options=user_list)
            
            if user_to_edit:
                edetails = auth_config['credentials']['usernames'][user_to_edit]
                with st.form(f"edit_form_{user_to_edit}"):
                    current_name_parts = edetails.get('name', '').split(' ', 1)
                    current_fname = current_name_parts[0] if len(current_name_parts) > 0 else ""
                    current_lname = current_name_parts[1] if len(current_name_parts) > 1 else ""
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        edit_fname = st.text_input(get_text('first_name', lang), value=current_fname)
                    with col2:
                        edit_lname = st.text_input(get_text('last_name', lang), value=current_lname)
                    edit_name = f"{edit_fname.strip()} {edit_lname.strip()}".strip()
                    
                    edit_email = st.text_input(get_text('email', lang), value=edetails.get('email', ''))
                    
                    role_options = {
                        get_text('role_viewer', lang): 'viewer',
                        get_text('role_admin', lang): 'admin'
                    }
                    
                    # Find current role index
                    current_r = edetails.get('role', 'viewer')
                    try:
                        r_idx = list(role_options.values()).index(current_r)
                    except:
                        r_idx = 0
                        
                    edit_role_label = st.selectbox(get_text('role', lang), options=list(role_options.keys()), index=r_idx)
                    edit_password = st.text_input(get_text('new_password_optional', lang), type="password")
                    
                    if st.form_submit_button(get_text('save_changes', lang)):
                        valid = True
                        if not edit_fname.strip() or not edit_lname.strip() or not edit_email:
                            st.error(get_text('fill_all_fields', lang))
                            valid = False
                        elif edit_password and len(edit_password) < 8:
                            st.error(get_text('password_length_error', lang))
                            valid = False
                            
                        if valid:
                            try:
                                auth_config['credentials']['usernames'][user_to_edit]['name'] = edit_name
                                auth_config['credentials']['usernames'][user_to_edit]['email'] = edit_email
                                auth_config['credentials']['usernames'][user_to_edit]['role'] = role_options[edit_role_label]
                                
                                if edit_password:
                                    auth_config['credentials']['usernames'][user_to_edit]['password'] = stauth.Hasher([edit_password]).generate()[0]
                                    
                                with open('auth_config.yaml', 'w', encoding='utf-8') as f:
                                    yaml.dump(auth_config, f, default_flow_style=False, sort_keys=False)
                                
                                st.success(get_text('user_updated_success', lang))
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error updating user: {e}")


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