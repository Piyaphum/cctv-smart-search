import cv2
from ultralytics import YOLO

# ==========================================
# ⚙️ ตั้งค่า (SETTINGS)
# ==========================================
# ใส่ชื่อไฟล์โมเดลที่คุณเทรนมา
MODEL_PATH = 'best.pt' 

# ใส่ชื่อไฟล์วิดีโอที่จะทดสอบ
VIDEO_PATH = 'mix.mp4'

# (ค่านี้สูงมาก จะจับเฉพาะช็อตที่ชัดเป๊ะๆ เท่านั้น)
CONFIDENCE = 0.85

# ==========================================
# 🚀 เริ่มทำงาน
# ==========================================
print(f"🔥 กำลังโหลดโมเดล: {MODEL_PATH} (Confidence Threshold: {CONFIDENCE}) ...")
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print("❌ หาไฟล์โมเดลไม่เจอ! อย่าลืมเอาไฟล์ best.pt มาวางไว้ที่เดียวกันนะ")
    exit()

cap = cv2.VideoCapture(VIDEO_PATH)

# ตั้งค่าบันทึกวิดีโอ
save_path = 'output_high_conf.avi'
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

print("🎥 เริ่มรันระบบ... กด 'q' เพื่อออก")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # ---------------------------------------------------------
    # 🧠 AI ทำงานตรงนี้
    # ---------------------------------------------------------
    # conf=CONFIDENCE (0.88) คือตัวกรองความมั่นใจ
    results = model.track(frame, persist=True, conf=CONFIDENCE, verbose=False)

    # 🎨 วาดกรอบ
    annotated_frame = results[0].plot()

    # แสดงผล
    cv2.imshow("YOLOv11 High Confidence", annotated_frame)
    out.write(annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"✅ เสร็จสิ้น! บันทึกไฟล์ผลลัพธ์ไว้ที่: {save_path}")