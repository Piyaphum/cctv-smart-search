# 🕵️‍♂️ CCTV Smart Search (Person Re-Identification)

ระบบค้นหาบุคคลเป้าหมายในกล้องวงจรปิดด้วย AI อัจฉริยะแบบ Multi-Target พร้อมระบบล็อคอิน (Authentication), ฐานข้อมูลเก็บบันทึกประวัติ (Database), และระบบแจ้งเตือนทางอีเมลอัตโนมัติ

## 🎯 ที่มาและจุดประสงค์ของระบบ
ในปัจจุบันการสืบค้นหาตัวบุคคลจากกล้องวงจรปิดหลายๆ ตัวใช้เวลานานและต้องใช้แรงงานคนในการนั่งดูวิดีโอ (Manual Review) ระบบนี้จึงถูกพัฒนาขึ้นเพื่อ **"ลดเวลาและเพิ่มความแม่นยำ"** ในการค้นหาบุคคลเป้าหมาย โดยใช้ AI วิเคราะห์ใบหน้า รูปร่าง และสีเสื้อผ้า และสามารถค้นหาเป้าหมายหลายคนได้พร้อมกันในคลิปเดียว เหมาะสำหรับงานรักษาความปลอดภัยภายในองค์กร หรือการสืบสวนคดีของตำรวจ

---

## 🛠️ Tech Stack (เทคโนโลยีที่ใช้)

โปรเจคนี้พัฒนาโดยใช้ภาษา **Python** 100% โดยแบ่งสถาปัตยกรรมออกเป็นส่วนๆ ดังนี้:

* **Frontend & Security (UI/UX & Auth):**
    * `Streamlit`: ใช้สร้าง Web Application Interface สไตล์ Dashboard ที่ใช้งานง่าย
    * `streamlit-authenticator`: จัดการระบบ Login, Logout และ Role-based access control (Admin / Viewer)
    * `bcrypt`: ใช้เข้ารหัส Password (Hashing) เพื่อความปลอดภัย
* **Database & Storage:**
    * `SQLite` (`sqlite3`): ฐานข้อมูลแบบ Local ฝังตัวเพื่อเก็บประวัติการค้นหา (`search_history`) และประวัติการตรวจพบ (`detections`)
    * `PyYAML`: ใช้โหลดและจัดการไฟล์ Configuration สำหรับ Users (`auth_config.yaml`)
* **Computer Vision & AI Core:**
    * `YOLOv8` (Ultralytics): โมเดลตรวจจับวัตถุ (Object Detection) ที่รวดเร็ว ใช้สำหรับตรวจจับ "มนุษย์" ในเฟรมวิดีโอ
    * `ResNet50` (PyTorch): โมเดล Deep Learning (Feature Extraction) เพื่อใช้ระบุตัวตนบุคคล (Re-ID) จากโครงสร้างร่างกาย
    * `OpenCV`: ใช้จัดการไฟล์วิดีโอ ตัดภาพตัวคน และประมวลผล Histogram ภาพเบื้องต้น
* **Data Processing & Utilities:**
    * `NumPy`, `Pandas`, `SciPy`: ใช้คำนวณทางคณิตศาสตร์, หาค่าความเหมือน (Cosine Similarity), และจัดการตารางข้อมูล
    * `SMTP`: ระบบส่งอีเมลแจ้งเตือนอัตโนมัติพร้อมรายงาน

---

## ⚙️ หลักการทำงาน (System Workflow)

ระบบมีการทำงานแบบ **Hybrid AI Matching** ซึ่งผสมผสานระหว่าง "รูปร่าง" และ "สีเสื้อ":

1. **Authentication (ยืนยันตัวตน):**
   * ผู้ใช้ต้อง Login ผ่านหน้าเว็บ หากรหัสผ่านถูกต้อง ระบบจะโหลดข้อมูลหน้า Dashboard ขึ้นมา
2. **Target Analysis (วิเคราะห์เป้าหมาย):**
   * อัปโหลดรูปเป้าหมาย 1 คนหรือมากกว่า ระบบจะใช้ **ResNet50** แปลงรูปภาพเป็น "เวกเตอร์ทางคณิตศาสตร์" (Embeddings)
   * คำนวณ "Histogram สีเสื้อ" ส่วนบนและทั้งตัว แยกต่างหาก
3. **Video Scanning (สแกนวิดีโอ):**
   * อ่านไฟล์วิดีโอและใช้ **YOLOv8** ตรวจจับคนในแต่ละเฟรม (ตาม Interval ที่กำหนด) และตัดภาพ (Crop) คนออกมา
4. **Feature Matching (การจับคู่แบบ Multi-Target):**
   * นำภาพคนที่เจอไปเทียบกับ **"เป้าหมายทุกคน"** ในฐานข้อมูลชั่วคราว
   * เปรียบเทียบด้วย *Cosine Similarity* (AI) และ *Histogram Correlation* (สีเสื้อ)
   * ดึงคนที่ได้คะแนนสูงสุดและผ่านเกณฑ์ Threshold
5. **Database Logging & Alert (เก็บข้อมูลและแจ้งเตือน):**
   * เมื่อประมวลผลวิดีโอจบ ระบบจะบันทึกผลลัพธ์ลงฐานข้อมูล SQLite (ใครค้นหาอะไร, เจอกี่ครั้ง)
   * ส่ง **Email** สรุปผลพร้อมจำนวนเป้าหมายที่พบแยกรายคน

---

## 💻 คู่มือการติดตั้ง (Installation Guide)

### 1. สิ่งที่ต้องมีก่อน (Prerequisites)
* **Python:** เวอร์ชัน 3.10 หรือ 3.11 แนะนำที่สุด (ไม่แนะนำ 3.13 ล่าสุดเพราะอาจมีปัญหากับบาง library)
* **Git:** สำหรับโคลนโปรเจค

### 2. ขั้นตอนการติดตั้ง (Step-by-Step)

**ขั้นตอนที่ 1: ดึงโค้ดจาก GitHub**
```bash
git clone https://github.com/Piyaphum/cctv-smart-search.git
cd cctv-smart-search
```

**ขั้นตอนที่ 2: สภาพแวดล้อม (Virtual Environment)**
```bash
python -m venv venv
venv\Scripts\activate   # สำหรับ Windows
# source venv/bin/activate # สำหรับ Mac/Linux
```

**ขั้นตอนที่ 3: ติดตั้ง Library**
```bash
pip install -r requirements.txt
```

*(หมายเหตุ: ระบบฐานข้อมูลและตารางจะถูกสร้างบรรจุในไฟล์ `cctv_search.db` อัตโนมัติเมื่อรันแอปครั้งแรก)*

---

## 🔐 การตั้งค่า Username / Password

ระบบเก็บข้อมูล Config รวมอยู่ในไฟล์ `auth_config.yaml` โดยรหัสผ่านต้องเป็น Hash ห้ามแก้เป็น Text ตรงๆ

**วิธีเปลี่ยน/เพิ่ม Password ใหม่:**
1. รันคำสั่งนี้ใน Terminal โดนเปลี่ยน `newpassword123` เป็นรหัสที่ต้องการ:
   ```bash
   python -c "import bcrypt; print(bcrypt.hashpw('newpassword123'.encode(), bcrypt.gensalt()).decode())"
   ```
2. นำผลลัพธ์ที่ได้ (ตัวหนังสือที่นำหน้าด้วย `$2b$12$...`) ไปใส่ในไฟล์ `auth_config.yaml` ตรงช่อง `password:` แทนที่ของเดิม

---

## 🚀 วิธีการใช้งาน (User Manual)

1. **รันโปรแกรม:** เปิด Terminal แล้วพิมพ์คำสั่ง
   ```bash
   streamlit run app.py
   ```

2. **Login เข้าใช้งาน และสิทธิ์การเข้าถึง (User Roles):**
   ระบบแบ่งผู้ใช้งานออกเป็น 2 ระดับ (Roles) ซึ่งมีสิทธิ์การเข้าถึงต่างกัน ดังนี้:

   * **👑 Admin (ผู้ดูแลระบบ)**
     * **หน้าที่:** บริหารจัดการและตรวจสอบภาพรวมของระบบ
     * **สิทธิ์การใช้งาน:**
       * ค้นหาบุคคลเป้าหมาย (Search Operation)
       * ดูผลลัพธ์ภาพที่ตรวจพบ (Result Gallery)
       * **เข้าถึงหน้า Admin Panel ได้ (เฉพาะ Admin เท่านั้น)**
       * สามารถดูภาพรวมสถิติการใช้งานทั้งระบบ (Total Searches, Total Detections, Active Users)
       * สามารถดู **ประวัติการค้นหาทั้งหมด (Search History)** ของผู้ใช้งานทุกคนในระบบ เพื่อการตรวจสอบ (Audit)
     * *ตัวอย่าง Login:* User: `admin` | Pass: `admin1234`

   * **👮 Viewer (เจ้าหน้าที่ปฏิบัติการ)**
     * **หน้าที่:** ผู้ปฏิบัติงานที่ใช้ระบบเพื่อค้นหาเป้าหมายเป็นหลัก
     * **สิทธิ์การใช้งาน:**
       * ค้นหาบุคคลเป้าหมาย (Search Operation)
       * ดูผลลัพธ์ภาพที่ตรวจพบ (Result Gallery) ของรอบนั้นๆ
       * ❌ **ไม่สามารถ** เข้าถึงหน้า Admin Panel ได้
       * ❌ **ไม่สามารถ** แอบดูประวัติการค้นหาของเจ้าหน้าที่คนอื่นๆ
     * *ตัวอย่าง Login:* User: `officer1` | Pass: `officer1234`

3. **กำหนดเป้าหมาย (Sidebar):** อัปโหลดภาพคนร้าย (ใส่ได้หลายภาพ)
4. **อัปโหลดวิดีโอ (Main):** อัปโหลดคลิปที่ต้องการสแกน ปรับค่า Threshold และกด "Start Multi-Search"
5. **ดูประวัติสำหรับแอดมิน:** หาก Login ด้วย Admin จะมีปุ่ม "Admin Panel" ที่ Sidebar สำหรับดูประวัติการสแกนและสถิติต่างๆ


---

## 🛠️ การแก้ไขเมื่อ Login ไม่ได้ (Troubleshooting)

หากเปิดหน้าเว็บขึ้นมาแล้ว **พิมพ์ Username/Password ถูกต้องแต่ระบบไม่ยอมล็อกอินเข้า หรือหน้าเว็บค้าง/หมุนโหลดไม่ยอมหยุด**:

1. **ตรวจสอบเว้นวรรคใน YAML:** ไปที่ไฟล์ `auth_config.yaml` ให้แน่ใจว่าการย่อหน้า (Indentation) ถูกต้อง และใช้ช่องว่าง (Space) ไม่ใช่ Tab
2. **รหัสผ่านไม่ถูกเข้ารหัส:** หากเผลอพิมพ์อักษรปกติลงในช่อง password จะ Error ให้รันสคริปต์ bcrypt ยามแปลง Hash ค่าพาสเวิร์ดมาใส่ใหม่ (ดูวิธีในหัวข้อการตั้งค่า Username)
3. **Cookie หมดอายุ หรือ ค้าง:**
   * ให้สลับไปใช้งาน Browser แบบไม่ระบุตัวตน (Incognito Mode) เพื่อทดสอบ
   * หรือล้าง Cookies ในเว็บเบราว์เซอร์สำหรับ `localhost`
4. **Python Version ของ Streamlit และ PyYAML ไม่ตรงกัน:** ลองใช้คำสั่งระบุเจาะจง `python -m pip install streamlit-authenticator bcrypt` แทนการใช้ `pip` ปกติ
