import streamlit as st
import cv2
import numpy as np
import tempfile
import torch
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights
from ultralytics import YOLO
from scipy.spatial.distance import cosine
from PIL import Image
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
import datetime
import glob
import shutil
import yaml
import streamlit_authenticator as stauth
from sklearn.cluster import KMeans
import webcolors

# --- ใหม่: นำเข้าฟังก์ชัน Database ของเรา ---
from database import init_db, log_search, get_history, save_target_profile, get_all_target_profiles, delete_target_profile

# เริ่มต้นฐานข้อมูล (สร้างตารางถ้ายังไม่มี)
init_db()

# --- CONFIG: ตั้งค่า URL ---
WEB_APP_URL = "http://localhost:8501"

# --- CONFIG: โฟลเดอร์เก็บผลลัพธ์ ---
RESULT_DIR = "detected_results"
os.makedirs(RESULT_DIR, exist_ok=True)

# --- CONFIG: Temp Dir ---
TEMP_DIR = "D:\\person-reid\\temp_video"
try:
    os.makedirs(TEMP_DIR, exist_ok=True)
except:
    pass 

# --- CONFIG: จำนวนรูปสูงสุดที่จะเก็บ (ป้องกัน Disk เต็ม) ---
MAX_IMAGES_KEPT = 100

# --- 1. AI Models Setup ---
@st.cache_resource
def load_models():
    detector = YOLO('yolov8n.pt')
    weights = ResNet50_Weights.DEFAULT
    reid_model = resnet50(weights=weights)
    reid_model.fc = torch.nn.Identity()
    reid_model.eval()
    base_transform = T.Compose([
        T.Resize((256, 128)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    aug_transform = T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        T.RandomRotation(degrees=10),
    ])
    return detector, reid_model, base_transform, aug_transform

detector, reid_model, base_transform, aug_transform = load_models()

# --- 2. Helper Functions ---
def extract_feature(image_pil, model, tf_func):
    img_tensor = tf_func(image_pil).unsqueeze(0)
    with torch.no_grad():
        feature = model(img_tensor).flatten().numpy()
    return feature

def get_part_histogram(image_pil, part='full'):
    img_np = np.array(image_pil)
    h, w, _ = img_np.shape
    if part == 'top': img_crop = img_np[:h//2, :]
    elif part == 'bottom': img_crop = img_np[h//2:, :]
    else: img_crop = img_np
    img_hsv = cv2.cvtColor(img_crop, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([img_hsv], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def closest_color(requested_color):
    """หาชื่อสีที่ใกล้เคียงที่สุดจาก RGB"""
    min_colors = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_color[0]) ** 2
        gd = (g_c - requested_color[1]) ** 2
        bd = (b_c - requested_color[2]) ** 2
        min_colors[(rd + gd + bd)] = name
    return min_colors[min(min_colors.keys())]

def get_dominant_color_name(image_pil):
    """สกัดสีเด่นจากเสื้อผ้า (ครึ่งบนของภาพ)"""
    try:
        img_np = np.array(image_pil)
        h, w, _ = img_np.shape
        # ตัดเอาแค่ 40% ด้านบน (ช่วงหน้าอก-ไหล่ ไม่เอาหน้า)
        top_crop = img_np[int(h*0.15):int(h*0.5), :]
        
        # Reshape สำหรับ KMeans
        pixels = top_crop.reshape(-1, 3)
        
        # รัน KMeans หา 2 สีหลัก
        kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
        kmeans.fit(pixels)
        
        # สีที่มีพื้นที่เยอะที่สุดใน 2 สี
        counts = np.bincount(kmeans.labels_)
        dominant = kmeans.cluster_centers_[np.argmax(counts)]
        
        # แปลง RGB เป็นชื่อสี
        color_name = closest_color([int(dominant[0]), int(dominant[1]), int(dominant[2])])
        return color_name.capitalize()
    except Exception as e:
        return "Unknown"

def generate_target_data(image_file, model, base_tf, aug_tf, n_aug=10):
    """สร้าง Embeddings สำหรับ Target 1 คน"""
    img = Image.open(image_file).convert('RGB')
    
    # Auto-crop ถ้ามีคนในรูป
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    results = detector(img_cv, classes=0, verbose=False)
    if len(results[0].boxes) > 0:
        box = results[0].boxes[0].xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = map(int, box)
        img = img.crop((x1, y1, x2, y2))

    embeddings = []
    hists_full = []
    hists_top = []
    
    def process_one(pil_img):
        embeddings.append(extract_feature(pil_img, model, base_tf))
        hists_full.append(get_part_histogram(pil_img, 'full'))
        hists_top.append(get_part_histogram(pil_img, 'top'))
        
    process_one(img)
    for _ in range(n_aug):
        aug_img = aug_tf(img)
        process_one(aug_img)
        
    return {
        "name": image_file.name,
        "image": img, # รูปต้นฉบับเอาไว้โชว์
        "embeddings": embeddings,
        "hists_full": hists_full,
        "hists_top": hists_top
    }

def clear_old_results():
    if os.path.exists(RESULT_DIR):
        shutil.rmtree(RESULT_DIR)
    os.makedirs(RESULT_DIR, exist_ok=True)

def maintain_storage_limit():
    """ลบรูปเก่าทิ้งถ้าเกินโควต้า"""
    all_files = []
    for root, dirs, files in os.walk(RESULT_DIR):
        for file in files:
            if file.endswith(".jpg"):
                all_files.append(os.path.join(root, file))
    
    if len(all_files) > MAX_IMAGES_KEPT:
        all_files.sort(key=os.path.getmtime)
        diff = len(all_files) - MAX_IMAGES_KEPT
        for i in range(diff):
            try: os.remove(all_files[i])
            except: pass

# --- 3. Email Function (Detailed Report) ---
def send_email_report(summary_dict, recipient_emails, sender_email, sender_password):
    try:
        msg = MIMEMultipart('alternative')
        msg['Subject'] = f'Security Alert: Found Matches in {len(summary_dict)} Videos'
        msg['From'] = sender_email
        msg['To'] = ", ".join(recipient_emails)
        
        # สร้างเนื้อหา HTML แยกตามวิดีโอ
        report_html = ""
        for video_name, targets in summary_dict.items():
            report_html += f"<h3>Video: {video_name}</h3><ul>"
            for target_name, logs in targets.items():
                count = len(logs)
                # ดึงสีเสื้อมาโชว์ในเมลด้วย (สุ่มมาสักสีหรือเอาสีแรก)
                color = logs[0]["color"] if logs else "Unknown"
                report_html += f"<li>⚠️ พบ <b>{target_name}</b> จำนวน <b>{count}</b> ครั้ง (👕 สีที่ใส่: {color})</li>"
            report_html += "</ul><hr>"

        html_body = f"""
        <html>
            <body style="font-family:Arial, sans-serif;">
                <h2 style="color:#d9534f;"> Security Alert Notification</h2>
                {report_html}
                <p>คลิกปุ่มด้านล่างเพื่อดูรูปภาพเหตุการณ์:</p>
                <a href="{WEB_APP_URL}" style="background-color:#4CAF50; color:white; padding:10px 20px; text-decoration:none; border-radius:5px;">
                    👉 ดูรูปภาพ (View Gallery)
                </a>
            </body>
        </html>
        """
        msg.attach(MIMEText(html_body, 'html'))
        
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
        return True, "Email sent successfully!"
    except Exception as e:
        return False, str(e)

# --- 4. Main UI ---
st.set_page_config(page_title="Multi-Target CCTV Search | KU Theme", layout="wide")

# ─────────────────────────────────────────────────────────────────
# 🔐 ระบบ LOGIN — ต้องผ่านก่อนถึงจะเข้าถึงหน้าหลักได้
# วิธีทำงาน:
#   1. โหลด users จากไฟล์ auth_config.yaml
#   2. สร้าง Authenticator object
#   3. แสดง login form — ถ้ายังไม่ login จะ stop ที่นี่
# ─────────────────────────────────────────────────────────────────
with open('auth_config.yaml', 'r', encoding='utf-8') as f:
    auth_config = yaml.safe_load(f)

authenticator = stauth.Authenticate(
    auth_config['credentials'],
    auth_config['cookie']['name'],
    auth_config['cookie']['key'],
    auth_config['cookie']['expiry_days']
)

# แสดง Login Form (ถ้ายังไม่ได้ login)
login_result = authenticator.login(location='main')

# ดึงสถานะ authentication จาก session_state
if not st.session_state.get('authentication_status'):
    if st.session_state.get('authentication_status') is False:
        st.error('❌ Username หรือ Password ไม่ถูกต้อง')
    else:
        st.info('👆 กรุณา Login ก่อนใช้งานระบบ')
    st.stop()  # ← หยุดแสดงโค้ดที่เหลือทั้งหมด ถ้ายังไม่ login

# ถ้า Login สำเร็จ — ดึงข้อมูล user ที่ login อยู่
current_user = st.session_state.get('username', 'unknown')
current_name = st.session_state.get('name', 'Unknown')
current_role = auth_config['credentials']['usernames'].get(current_user, {}).get('role', 'viewer')

# แสดงปุ่ม Logout และชื่อ User ใน Sidebar
with st.sidebar:
    st.markdown(f"### 👤 {current_name}")
    st.caption(f"Role: `{current_role}`")
    authenticator.logout('Logout', 'sidebar')
    st.divider()

# Custom CSS for KU Theme
st.markdown("""
    <style>
        /* Primary Colors */
        :root {
            --ku-green: #006664;
            --ku-fresh-green: #B2BB1E;
        }
        
        .main {
            background-color: #f8f9fa;
        }
        
        /* Headers */
        h1, h2, h3, h4 {
            color: var(--ku-green) !important;
            font-family: 'Inter', sans-serif;
        }
        
        /* Sidebar */
        [data-testid="stSidebar"] {
            background-color: #ffffff;
            border-right: 1px solid #e0e0e0;
        }
        
        /* Buttons */
        .stButton>button {
            border-radius: 8px !important;
            transition: all 0.3s ease !important;
            font-weight: 600 !important;
        }
        
        .stButton>button[kind="primary"] {
            background-color: var(--ku-green) !important;
            border: none !important;
        }
        
        .stButton>button[kind="primary"]:hover {
            background-color: var(--ku-fresh-green) !important;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1) !important;
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 6px;
            background-color: transparent;
        }
        
        /* Inactive Tab: เขียวอ่อน ข้อความเข้ม */
        .stTabs [data-baseweb="tab"] {
            background-color: #d4edec !important;
            border-radius: 8px 8px 0 0 !important;
            padding: 10px 22px !important;
            border: 2px solid #a8d5d3 !important;
            border-bottom: none !important;
            color: #003d3c !important;
            font-weight: 700 !important;
            font-size: 15px !important;
        }
        
        /* Inactive Tab Hover */
        .stTabs [data-baseweb="tab"]:hover {
            background-color: #b5dedd !important;
            color: #002828 !important;
        }
        
        /* Active Tab: เขียวเกษตรเข้ม ข้อความขาว */
        .stTabs [aria-selected="true"] {
            background-color: var(--ku-green) !important;
            color: white !important;
            border: 2px solid var(--ku-green) !important;
            border-bottom: none !important;
        }
        
        /* Tab panel border */
        .stTabs [data-baseweb="tab-panel"] {
            border: 2px solid #a8d5d3;
            border-radius: 0 8px 8px 8px;
            padding: 16px;
        }

        /* User Manual Floating Button */
        #manual-btn {
            position: fixed;
            bottom: 30px;
            left: 30px;
            z-index: 9999;
            background-color: var(--ku-green);
            color: white !important;
            padding: 12px 24px;
            border-radius: 30px;
            text-decoration: none;
            font-weight: bold;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            transition: all 0.3s;
            border: 2px solid white;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        #manual-btn:hover {
            background-color: var(--ku-fresh-green);
            transform: translateY(-3px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.3);
        }
    </style>
    <a href="#user-manual" id="manual-btn">
        📖 คู่มือการใช้งาน
    </a>
    """, unsafe_allow_html=True)

st.title("CCTV Search")

tab1, tab2 = st.tabs(["🎥 Search Operation", "📂 Result Gallery"])

# --- TAB 1: Search ---
with tab1:
    col_sidebar, col_main = st.columns([1, 2])
    
    with col_sidebar:
        st.header("1. Target Config")
        
        targets_db = [] # เก็บข้อมูล Target ทุกคนที่เลือก
        
        # --- TAB ย่อยสำหรับจัดการ Target ---
        t_tab1, t_tab2 = st.tabs(["🗃️ Load from DB", "➕ New Upload"])
        
        with t_tab1:
            saved_targets = get_all_target_profiles()
            if saved_targets:
                st.caption("เลือกคนที่เคยบันทึกไว้แล้ว:")
                for tgt in saved_targets:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        if st.checkbox(f"🧑 {tgt['name']}", key=f"sel_{tgt['id']}"):
                            # ถ้าติ๊กเลือก ให้แปลงกลับเป็นรูปแบบที่ระบบ search เข้าใจ
                            targets_db.append({
                                "name": tgt["name"],
                                "image": None, # รูปไม่มีเพราะโหลดจาก DB คืน
                                "embeddings": tgt["embeddings"],
                                "hists_full": tgt["hists_full"],
                                "hists_top": tgt["hists_top"]
                            })
                    with col2:
                        if st.button("🗑️", key=f"del_{tgt['id']}", help="ลบประวัติคนนี้"):
                            delete_target_profile(tgt['id'])
                            st.rerun()
            else:
                st.info("ยังไม่มีข้อมูลคนร้ายในระบบ")

        with t_tab2:
            target_files = st.file_uploader("Upload Targets (1 or more)", type=['jpg', 'png'], accept_multiple_files=True)
            save_to_db = st.checkbox("💾 Save to Database", value=True, help="บันทึกข้อมูลหน้าตาและสีเสื้อไว้ค้นหาครั้งหน้า")
            
            if target_files:
                with st.spinner("Processing New Targets..."):
                    for t_file in target_files:
                        # สร้างชื่อใหม่ถ้ามีช่อง Save Target
                        target_name = st.text_input(f"ตั้งชื่อสำหรับ: {t_file.name}", value=t_file.name.split('.')[0])
                        
                        t_data = generate_target_data(t_file, reid_model, base_transform, aug_transform)
                        t_data["name"] = target_name # อัปเดตชื่อ
                        targets_db.append(t_data)
                        
                        # โชว์รูป
                        c1, c2 = st.columns([1, 3])
                        c1.image(t_data['image'], use_container_width=True)
                        c2.caption(f"✅ {target_name}")

                        # บันทึกลงฐานข้อมูลถ้าติ๊กถูก
                        if save_to_db:
                            if st.button(f"Save '{target_name}' to DB", key=f"btn_save_{target_name}"):
                                save_target_profile(
                                    name=target_name,
                                    embeddings=t_data["embeddings"],
                                    hists_full=t_data["hists_full"],
                                    hists_top=t_data["hists_top"],
                                    created_by=current_user
                                )
                                st.success("Saved!")
        
        st.success(f"Ready: {len(targets_db)} Targets Selected")
                        
        st.divider()
        enable_email = st.checkbox("Email Report?", value=True)
        recipient_emails = []
        if enable_email:
            num_recipients = int(st.number_input("Number of recipients", min_value=1, step=1, value=1))
            for i in range(num_recipients):
                r_email = st.text_input(f"Recipient {i+1} Email", key=f"recipient_email_{i}")
                if r_email:
                    recipient_emails.append(r_email)
        
        # ⚠️ อย่าลืมแก้ตรงนี้
        sender_email = "piyaphum1492@gmail.com" 
        sender_password = "vhvp varc qflt ryxv"

    with col_main:
        st.header("2. Video Scanning")
        video_files = st.file_uploader("Upload CCTV Videos", type=['mp4', 'avi'], accept_multiple_files=True)
        
        c1, c2, c3 = st.columns(3)
        threshold = c1.slider("Threshold", 0.0, 1.0, 0.70)
        shirt_strictness = c2.slider("Shirt Strictness", 0.0, 1.0, 0.6)
        snapshot_interval = c3.slider("Snapshot (sec)", 0.5, 5.0, 1.0) 

        if st.button("Start Multi-Search", type="primary") and video_files and targets_db:
            
            clear_old_results()
            
            # ตัวแปรเก็บสรุปผลเพื่อส่งเมล { "video1.mp4": {"target1": 5, "target2": 0}, ... }
            report_summary = {} 
            
            total_videos = len(video_files)
            main_progress = st.progress(0)
            status_text = st.empty()
            
            for v_idx, video_file in enumerate(video_files):
                video_name = video_file.name
                report_summary[video_name] = {} # เริ่มต้นนับของวิดีโอนี้
                
                status_text.write(f"Processing: **{video_name}**")
                
                with st.expander(f"Monitoring: {video_name}", expanded=True):
                    # Save Video to Temp
                    tfile_path = os.path.join(TEMP_DIR, f"temp_{video_name}")
                    try:
                        with open(tfile_path, "wb") as f:
                            while True:
                                chunk = video_file.read(4*1024*1024)
                                if not chunk: break
                                f.write(chunk)
                        
                        cap = cv2.VideoCapture(tfile_path)
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        
                        cols = st.columns(3)
                        last_snap = -snapshot_interval
                        frame_idx = 0
                        found_count = 0
                        
                        while cap.isOpened():
                            ret, frame = cap.read()
                            if not ret: break
                            frame_idx += 1
                            
                            if total_frames > 0 and frame_idx % 50 == 0:
                                prog = (v_idx + (frame_idx/total_frames))/total_videos
                                main_progress.progress(min(prog, 1.0))
                            
                            curr_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                            if (curr_time - last_snap) < snapshot_interval:
                                continue
                            last_snap = curr_time
                            
                            # Detect People
                            results = detector(frame, classes=0, verbose=False)
                            for r in results:
                                boxes = r.boxes
                                for box in boxes:
                                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                                    if (x2-x1) < 40 or (y2-y1) < 80: continue
                                    
                                    person_crop = frame[y1:y2, x1:x2]
                                    if person_crop.size == 0: continue
                                    person_pil = Image.fromarray(cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB))
                                    
                                    curr_emb = extract_feature(person_pil, reid_model, base_transform)
                                    curr_hist_full = get_part_histogram(person_pil, 'full')
                                    curr_hist_top = get_part_histogram(person_pil, 'top')
                                    
                                    # --- Multi-Target Matching Logic ---
                                    best_match_target = None
                                    highest_score = 0.0
                                    
                                    # วนลูปเทียบกับ Target ทุกคน
                                    for t_data in targets_db:
                                        # เทียบกับ Augmented References ของ Target นั้นๆ
                                        for t_emb, t_full, t_top in zip(t_data['embeddings'], t_data['hists_full'], t_data['hists_top']):
                                            ai_score = 1 - cosine(t_emb, curr_emb)
                                            full_score = max(0, cv2.compareHist(t_full, curr_hist_full, cv2.HISTCMP_CORREL))
                                            shirt_score = max(0, cv2.compareHist(t_top, curr_hist_top, cv2.HISTCMP_CORREL))
                                            
                                            total = (ai_score * 0.5) + ((full_score*0.4 + shirt_score*0.6) * 0.5)
                                            if shirt_score < 0.5: total -= (shirt_strictness * 0.5)
                                            
                                            if total > highest_score:
                                                highest_score = total
                                                best_match_target = t_data['name'] # จำชื่อคนที่มีคะแนนสูงสุด
                                    
                                    # ถ้าคะแนนสูงสุด ผ่านเกณฑ์
                                    if highest_score > threshold:
                                        found_count += 1
                                        
                                        # 🎨 ดึงชื่อสีเสื้อผ้าออกมา
                                        color_name = get_dominant_color_name(person_pil)
                                        
                                        # อัปเดตยอดสรุป สำหรับส่งอีเมล
                                        if best_match_target not in report_summary[video_name]:
                                            report_summary[video_name][best_match_target] = []
                                        # เก็บ score กับ เวลา เข้าไปด้วย
                                        report_summary[video_name][best_match_target].append({
                                            "score": highest_score,
                                            "color": color_name
                                        })
                                        
                                        # Save Image (แยกโฟลเดอร์ตามวิดีโอ)
                                        video_result_dir = os.path.join(RESULT_DIR, video_name)
                                        os.makedirs(video_result_dir, exist_ok=True)
                                        
                                        timestamp_str = datetime.datetime.now().strftime("%H%M%S_%f")
                                        # ใส่สีเข้าไปในชื่อไฟล์ด้วย Found_TargetName_Color_...
                                        save_name = f"Found_{best_match_target}_{color_name}_{timestamp_str}_{highest_score:.2f}.jpg"
                                        person_pil.save(os.path.join(video_result_dir, save_name))
                                        
                                        maintain_storage_limit()
                                        
                                        # แสดงผลหน้าเว็บ
                                        cols[found_count % 3].image(person_pil, caption=f"{best_match_target}\n🕒 {curr_time:.1f}s | 🎯 {highest_score * 100:.0f}%\n👕 Color: {color_name}")

                        cap.release()
                    except Exception as e:
                        st.error(f"Error: {e}")
                    finally:
                        if os.path.exists(tfile_path):
                            try: os.remove(tfile_path)
                            except: pass
            
            main_progress.progress(1.0)
            status_text.success("✅ All Done!")
            
            # เช็คว่าเจอใครบ้างมั้ย
            total_found = sum([len(v) for vid_dict in report_summary.values() for v in vid_dict.values()])
            
            # ─────────────────────────────────────────────────────────
            # 💾 บันทึก Search History ลง Database
            # วนลูปผลลัพธ์ทุกวิดีโอ และทุก target แล้วบันทึกลง DB
            # ─────────────────────────────────────────────────────────
            for vid_name, targets in report_summary.items():
                for tgt_name, logs in targets.items():
                    log_search(
                        username=current_user,
                        video_name=vid_name,
                        target_name=tgt_name,
                        total_found=len(logs)
                    )
            if total_found > 0:
                st.toast(f"💾 บันทึกผลลัพธ์ลง Database แล้ว", icon="🗄️")
            
            if total_found > 0:
                st.success(f"Found {total_found} matches total.")
                if enable_email and len(recipient_emails) > 0:
                    with st.spinner("Sending Report..."):
                        success, msg = send_email_report(report_summary, recipient_emails, sender_email, sender_password)
                        if success: st.toast("Email Sent!", icon="📧")
                        else: st.error(msg)
                elif enable_email:
                    st.warning("No recipient emails provided.")
            else:
                st.warning("No matches found.")

# --- TAB 2: Gallery ---
with tab2:
    st.header("📂 Detection Gallery (Grouped by Video)")
    if st.button("🔄 Refresh"):
        st.rerun()
    
    # วนลูปดูโฟลเดอร์ย่อย (แต่ละวิดีโอ)
    if os.path.exists(RESULT_DIR):
        video_folders = [f for f in os.listdir(RESULT_DIR) if os.path.isdir(os.path.join(RESULT_DIR, f))]
        
        if not video_folders:
            st.info("No results yet.")
        
        for vid_folder in video_folders:
            with st.expander(f"📁 {vid_folder}", expanded=True):
                folder_path = os.path.join(RESULT_DIR, vid_folder)
                images = glob.glob(os.path.join(folder_path, "*.jpg"))
                images.sort(key=os.path.getmtime, reverse=True)
                
                if images:
                    cols = st.columns(5)
                    for i, img_path in enumerate(images):
                        img = Image.open(img_path)
                        fname = os.path.basename(img_path)
                        # แกะชื่อ Target และ สี ออกจากรูป (Found_TargetName_ColorName_...)
                        parts = fname.split('_')
                        if len(parts) >= 3:
                            display_name = parts[1]
                            color_name = parts[2]
                            cols[i % 5].image(img, caption=f"🧑 {display_name}\n👕 {color_name}", use_container_width=True)
                        else:
                            # fallback ของเก่า
                            display_name = parts[1] if len(parts)>1 else "Unknown"
                            cols[i % 5].image(img, caption=f"{display_name}", use_container_width=True)
                else:
                    st.caption("No images.")
    else:
        st.warning("Results directory not found.")

# --- 5. User Manual Section ---
st.divider()
st.markdown("<div id='user-manual'></div>", unsafe_allow_html=True)
with st.expander("คู่มือการใช้งาน (User Manual)", expanded=False):
    st.write("""
    ### วิธีใช้งานระบบ Multi-Target CCTV Search (KU Edition)
    
    ยินดีต้อนรับสู่ระบบค้นหาบุคคลจากกล้องวงจรปิดธีม ม.เกษตร!
    
    #### ขั้นตอนการใช้งาน:
    1. **อัปโหลดภาพเป้าหมาย (Target Config):** 
       - อัปโหลดรูปภาพใบหน้าหรือตัวบุคคลที่ต้องการค้นหา (รองรับหลายคนพร้อมกัน)
       - ระบบจะสร้าง Embedding และ Histogram เพื่อใช้ในการเปรียบเทียบ
    
    2. **ตั้งค่าการแจ้งเตือน (Email Report):**
       - ติ๊กถูกที่ช่อง 'Email Report?' 
       - กรอกอีเมลผู้รับเพื่อรับรายงานสรุปผลการตรวจพบ
    
    3. **อัปโหลดวิดีโอ (Video Scanning):**
       - เลือกไฟล์วิดีโอ CCTV (mp4, avi) ที่ต้องการสแกน
    
    4. **ปรับตั้งค่าการค้นหา:**
       - **Threshold:** ค่าความแม่นยำในการจับคู่ (แนะนำ 0.70)
       - **Shirt Strictness:** ความเข้มงวดในการเทียบสีเสื้อ (ช่วยลด False Positive)
       - **Snapshot Interval:** ความถี่ในการดึงภาพจากวิดีโอมาวิเคราะห์
    
    5. **เริ่มการค้นหา:**
       - กดปุ่ม **Start Multi-Search** และรอระบบประมวลผล
    
    6. **ตรวจสอบผลลัพธ์:**
       - ผลลัพธ์ที่ตรวจพบจะแสดงแบบ Real-time และถูกเก็บไว้ในแท็บ **📂 Result Gallery**
    
    ---
    *หากพบปัญหาการใช้งาน ติดต่อผู้ดูแลระบบ*
    """)
