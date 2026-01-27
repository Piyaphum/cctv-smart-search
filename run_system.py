import cv2
import numpy as np
from ultralytics import YOLO

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
VIDEO_PATH = 'mix.mp4'          
MODEL_PATH = 'yolov11m.pt'      
CONF_THRESHOLD = 0.5            

# เกณฑ์ความดำ (ปรับแค่ตัวนี้ตัวเดียว)
# 0.30 หมายถึง ในพื้นที่ตรงกลางต้องมีสีดำ 30%
BLACK_THRESHOLD = 0.30          

# ==========================================
# LOGIC: BLACK SHIRT CHECK
# ==========================================
def is_black_outfit(img):
    h, w, _ = img.shape
    if h < 50: return False # ตัวเล็กไป ข้าม

    # --- 1. เจาะไข่แดง (ดูเฉพาะกลางหน้าอก) ---
    # ตัดบน 20% (ตัดหัว), ตัดล่าง 40% (ตัดขา)
    # ตัดซ้าย 30%, ตัดขวา 30% (หลบกำแพง/ประตู)
    y1 = int(h * 0.20)
    y2 = int(h * 0.60)
    x1 = int(w * 0.30)
    x2 = int(w * 0.70)
    
    center_part = img[y1:y2, x1:x2]
    
    if center_part.size == 0: return False

    # --- 2. เช็คสีดำ (HSV) ---
    hsv = cv2.cvtColor(center_part, cv2.COLOR_BGR2HSV)
    
    # นิยามสีดำ: V (ความสว่าง) ต้องน้อยกว่า 130
    # (เพิ่มให้หน่อยเผื่อแสงไฟสว่าง)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 130])
    
    mask = cv2.inRange(hsv, lower_black, upper_black)
    black_ratio = np.count_nonzero(mask) / center_part.size

    # ถ้าดำเกินเกณฑ์ -> ใช่เลย
    return black_ratio > BLACK_THRESHOLD

# ==========================================
# 🚀 MAIN LOOP
# ==========================================
try:
    model = YOLO(MODEL_PATH)
except:
    model = YOLO('yolov8n.pt')

cap = cv2.VideoCapture(VIDEO_PATH)
width = int(cap.get(3))
height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter('output_black_shirt.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

print("🎥 เริ่มจับคนเสื้อดำ... (Black Shirt Only)")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # Track คน
    results = model.track(frame, persist=True, tracker="bytetrack.yaml", classes=[0], verbose=False, conf=CONF_THRESHOLD)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)

        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = map(int, box)
            
            # Crop ภาพคน
            person_img = frame[max(0,y1):min(height,y2), max(0,x1):min(width,x2)]
            
            # เช็คว่าเสื้อดำไหม?
            if is_black_outfit(person_img):
                # 🎯 เจอเสื้อดำ! (แดง)
                color = (0, 0, 255) 
                label = f"TARGET {track_id}"
                thickness = 4
            else:
                # 🍃 เสื้อสีอื่น (เขียว)
                color = (0, 255, 0)
                label = f"ID {track_id}"
                thickness = 2

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(frame, label, (x1, y1-10), 0, 0.7, color, 2)

    cv2.imshow("Simple Black Detector", frame)
    out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
out.release()
cv2.destroyAllWindows()
print("เสร็จสิ้น")