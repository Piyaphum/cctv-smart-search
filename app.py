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
import pandas as pd
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from email.mime.image import MIMEImage
import io
import os

# --- CONFIG: ยังคงให้ใช้ Drive D พักไฟล์ เพื่อกันคอมค้าง ---
TEMP_DIR = "D:\\person-reid\\temp_video"
try:
    os.makedirs(TEMP_DIR, exist_ok=True)
except:
    pass # ถ้าสร้างไม่ได้ เดี๋ยวระบบจะไปใช้ Default Temp เอง

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

# --- 2. Helper Functions (AI Logic) ---
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

def generate_augmented_references(original_img_pil, model, base_tf, aug_tf, n_aug=10):
    embeddings = []
    hists_full = []
    hists_top = []
    def process_one(img):
        embeddings.append(extract_feature(img, model, base_tf))
        hists_full.append(get_part_histogram(img, 'full'))
        hists_top.append(get_part_histogram(img, 'top'))
    process_one(original_img_pil)
    for _ in range(n_aug):
        aug_img = aug_tf(original_img_pil)
        process_one(aug_img)
    return embeddings, hists_full, hists_top

# --- 3. Email Function ---
def send_email_report(log_data, recipient_email, sender_email, sender_password):
    try:
        msg = MIMEMultipart('related')
        msg['Subject'] = f'🚨 Security Alert: Person Found in {len(set(d["Video"] for d in log_data))} Videos'
        msg['From'] = sender_email
        msg['To'] = recipient_email
        
        sorted_log = sorted(log_data, key=lambda x: x['Similarity_Score'], reverse=True)[:5]
        
        html_rows = ""
        images_to_attach = []
        for i, row in enumerate(sorted_log):
            cid = f"image_{i}"
            score_color = "#e6ffe6" if row['Similarity_Score'] > 0.8 else "#ffffff"
            html_rows += f"""<tr style="background-color: {score_color};"><td style="padding:10px;">{i+1}</td><td style="padding:10px;"><b>{row['Video']}</b><br>@ {row['Timestamp_Sec']}s</td><td style="padding:10px;">{row['Similarity_Score']:.4f}</td><td style="padding:10px;">{row['Shirt_Match_Score']:.4f}</td><td style="padding:10px;"><img src="cid:{cid}" width="100" style="border-radius:5px;"></td></tr>"""
            images_to_attach.append((cid, row['Image_PIL']))
            
        html_body = f"""<html><body style="font-family:Arial;"><h2>📊 CCTV Finding Report</h2><p>พบเป้าหมายทั้งหมด <b>{len(log_data)}</b> ครั้ง</p><table style="width:100%;border-collapse:collapse;"><thead><tr style="background:#f2f2f2;"><th>#</th><th>Video / Time</th><th>Score</th><th>Shirt Score</th><th>Snapshot</th></tr></thead><tbody>{html_rows}</tbody></table></body></html>"""
        
        msg_alternative = MIMEMultipart('alternative')
        msg.attach(msg_alternative)
        msg_alternative.attach(MIMEText(html_body, 'html'))
        
        for cid, img_pil in images_to_attach:
            img_byte_arr = io.BytesIO()
            img_pil.save(img_byte_arr, format='JPEG')
            img_data = img_byte_arr.getvalue()
            image_part = MIMEImage(img_data)
            image_part.add_header('Content-ID', f'<{cid}>')
            image_part.add_header('Content-Disposition', 'inline', filename=f'{cid}.jpg')
            msg.attach(image_part)
            
        df = pd.DataFrame([{k: v for k, v in d.items() if k != 'Image_PIL'} for d in log_data])
        csv_data = df.to_csv(index=False).encode('utf-8')
        part = MIMEApplication(csv_data, Name='full_report.csv')
        part['Content-Disposition'] = 'attachment; filename="full_report.csv"'
        msg.attach(part)
        
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
        return True, "Email sent successfully!"
    except Exception as e:
        return False, str(e)

# --- 4. Main UI ---
st.set_page_config(page_title="Simple CCTV Search", layout="wide")
st.title("🕵️‍♂️ CCTV Smart Search (Simple Mode)")

col_sidebar, col_main = st.columns([1, 2])

with col_sidebar:
    st.header("1. Target & Email")
    ref_file = st.file_uploader("Upload Target Person", type=['jpg', 'png'])
    st.divider()
    enable_email = st.checkbox("Enable Email Report?")
    recipient_email = st.text_input("Recipient Email")
    # Credentials (Hardcoded for convenience)
    sender_email = "piyaphum1492@gmail.com" 
    sender_password = "gite wrcl rtkg iyca" 

    target_embeddings = []
    target_hists_full = []
    target_hists_top = []

    if ref_file:
        ref_img = Image.open(ref_file).convert('RGB')
        st.image(ref_img, caption="Target", use_container_width=True)
        ref_cv = cv2.cvtColor(np.array(ref_img), cv2.COLOR_RGB2BGR)
        results = detector(ref_cv, classes=0, verbose=False)
        target_crop_pil = ref_img
        if len(results[0].boxes) > 0:
            box = results[0].boxes[0].xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, box)
            target_crop_pil = ref_img.crop((x1, y1, x2, y2))

        with st.spinner("Processing Target..."):
            target_embeddings, target_hists_full, target_hists_top = generate_augmented_references(
                target_crop_pil, reid_model, base_transform, aug_transform, n_aug=10
            )
        st.success("Target Ready ✅")

with col_main:
    st.header("2. Video Upload")
    video_files = st.file_uploader("Upload Videos (Multiple allowed)", 
                                 type=['mp4', 'avi'], 
                                 accept_multiple_files=True)
    
    c1, c2 = st.columns(2)
    threshold = c1.slider("AI Threshold", 0.0, 1.0, 0.70)
    shirt_strictness = c2.slider("Shirt Color Strictness", 0.0, 1.0, 0.6)

    if st.button("🚀 Start Search", type="primary") and video_files and len(target_embeddings) > 0:
        
        all_videos_log = []
        total_videos = len(video_files)
        main_progress = st.progress(0)
        status_text = st.empty()
        
        for v_idx, video_file in enumerate(video_files):
            video_name = video_file.name
            status_text.write(f"🎞️ Processing: **{video_name}** ({v_idx+1}/{total_videos})")
            
            with st.expander(f"Result: {video_name}", expanded=True):
                # --- [Safe Save] บันทึกไฟล์ลง Drive D แบบ Chunk (แก้คอมค้าง) ---
                tfile_path = os.path.join(TEMP_DIR, f"temp_{video_name}")
                try:
                    with open(tfile_path, "wb") as f:
                        while True:
                            chunk = video_file.read(4 * 1024 * 1024)
                            if not chunk: break
                            f.write(chunk)
                    
                    cap = cv2.VideoCapture(tfile_path)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    
                    col1, col2, col3 = st.columns(3)
                    found_count = 0
                    frame_idx = 0
                    
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret: break
                        frame_idx += 1
                        
                        if total_frames > 0 and frame_idx % 50 == 0:
                            prog = (v_idx + (frame_idx / total_frames)) / total_videos
                            main_progress.progress(min(prog, 1.0))
                        
                        if frame_idx % 10 != 0: continue
                        
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
                                
                                best_score = 0.0
                                for t_emb, t_full, t_top in zip(target_embeddings, target_hists_full, target_hists_top):
                                    ai_score = 1 - cosine(t_emb, curr_emb)
                                    full_score = max(0, cv2.compareHist(t_full, curr_hist_full, cv2.HISTCMP_CORREL))
                                    shirt_score = max(0, cv2.compareHist(t_top, curr_hist_top, cv2.HISTCMP_CORREL))
                                    total = (ai_score * 0.5) + ((full_score*0.4 + shirt_score*0.6) * 0.5)
                                    if shirt_score < 0.5: total -= (shirt_strictness * 0.5)
                                    if total > best_score: best_score = total
                                
                                if best_score > threshold:
                                    found_count += 1
                                    timestamp_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                                    
                                    entry = {
                                        "Video": video_name,
                                        "Timestamp_Sec": round(timestamp_sec, 2),
                                        "Similarity_Score": round(best_score, 4),
                                        "Shirt_Match_Score": round(shirt_score, 4),
                                        "Image_PIL": person_pil
                                    }
                                    all_videos_log.append(entry)
                                    
                                    with col1 if found_count % 3 == 1 else col2 if found_count % 3 == 2 else col3:
                                        st.image(person_pil, caption=f"{timestamp_sec:.1f}s | {best_score:.2f}")

                    cap.release()
                    st.info(f"Done. Found {found_count} matches.")
                
                except Exception as e:
                    st.error(f"Error: {e}")
                finally:
                    # ลบไฟล์ Temp ทิ้งเพื่อคืนพื้นที่
                    if os.path.exists(tfile_path):
                        try: os.remove(tfile_path)
                        except: pass

        main_progress.progress(1.0)
        status_text.success("✅ Process Completed!")
        
        if all_videos_log:
            st.divider()
            st.metric("Total Matches Found", len(all_videos_log))
            
            if enable_email:
                with st.spinner("📧 Sending Email Report..."):
                    success, msg = send_email_report(all_videos_log, recipient_email, sender_email, sender_password)
                    if success: st.toast("Email Sent Successfully!"git add ., icon="📧")
                    else: st.error(msg)
        else:

            st.warning("No matches found in any video.")