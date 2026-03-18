# 📝 บันทึกการอัปเดต (CHANGELOG)

## ลำดับเวอร์ชันการพัฒนา

### [v2.0.0] - 2026-03-18 🚀 **Major Refactoring & Multilingual Support**

#### ✨ ฟีเจอร์ใหม่
- **📱 ระบบเลือกภาษา (Language Support)**
  - ปุ่มเลือกภาษาในแถบด้านข้าง (Sidebar)
  - รองรับภาษาไทยและอังกฤษทั้งระบบ
  - เปลี่ยนภาษาได้ทันทีโดยไม่ต้องรีโหลดหน้า

- **❓ ระบบช่วยเหลือ (Tooltip/Help Text)**
  - ทุก Parameter slider มีไอคอน `?` ด้านบน
  - Hover บนตัวเลือกจะแสดงคำอธิบายโดยละเอียด
  - ช่วยผู้ใช้ทำความเข้าใจการใช้งาน

#### 🏗️ การปรับปรุงสถาปัตยกรรม (Architecture Refactoring)
- **แยกไฟล์เป็น Modules แยกหมวดหมู่**
  - `app.py` - Main application (~280 บรรทัด เรียบร้อย)
  - `config.py` - ค่าคงที่และการตั้งค่า
  - `models.py` - โหลด AI models
  - `feature_extraction.py` - สกัด embeddings, สี, เพศ
  - `target_management.py` - จัดการ target profiles
  - `video_processor.py` - ประมวลผลวิดีโอ
  - `search_engine.py` - ตรรกะการจับคู่
  - `email_service.py` - ส่งอีเมลแจ้งเตือน
  - `ui_styles.py` - สไตล์ UI (ถูกลบแล้ว)
  - `translations.py` - ข้อความแปลภาษา

- **ข้อดีของการแยก:**
  - โค้ดอ่านง่ายขึ้น ไม่รกอีกต่อไป
  - เปลี่ยนแปลงฟีเจอร์ใหม่ได้ง่ายขึ้น
  - สามารถเพิ่มจำนวน modules ลงไป

#### 🎨 ปรับปรุง UI/UX
- **เปลี่ยน Theme ของ Streamlit**
  - ปล่อยให้ใช้ Default Streamlit Dark Theme
  - ปุ่มและ sliders มองเห็นชัดเจน
  - ลบการปรับแต่ง CSS ที่ซับซ้อน

- **ปรับปรุงรูปแบบข้อความ**
  - ใช้ภาษาท้องถิ่น (ไทย/อังกฤษ)
  - ป้ายกำกับชัดเจน
  - เนื้อหาคำอธิบายครบถ้วน

#### 📚 เอกสาร (Documentation)
- เพิ่ม `ARCHITECTURE.md` - รายละเอียดโครงสร้างโปรเจค
- เพิ่ม `translations.py` - ข้อความแปล
- สร้าง `.streamlit/config.toml` - ตั้งค่า Streamlit

#### 🔧 การแก้ไข
- ลบ import ที่ไม่ใช้
- เรียงลำดับ imports ให้ถูกต้อง
- แก้ไข syntax errors

---

### [v1.5.0] - 2026-03-15 🎯 **Admin Panel & Advanced Features**

#### ✨ ฟีเจอร์ใหม่
- **👑 Admin Panel**
  - สถิติการใช้งาน (Total Searches, Detections, Active Users)
  - ประวัติการค้นหาทั้งหมด (Search History)
  - system Clean-up ลบไฟล์ขยะ

- **📊 ระบบสถิติ**
  - นับจำนวนการค้นหา
  - นับจำนวนการตรวจพบ
  - แสดงผู้ใช้งานที่เคยใช้

- **🗂️ ระบบกรองประวัติ**
  - กรองตามวันที่
  - กรองตามชื่อผู้ใช้งาน

#### 🐛 Bug Fixes
- แก้ไขปัญหาการบันทึกลงฐานข้อมูล
- ปรับปรุงความเสถียรของระบบ

---

### [v1.0.0] - 2026-03-01 🎉 **Initial Release**

#### ✨ ฟีเจอร์หลัก
- **👥 ระบบ Authentication**
  - Login/Logout ปลอดภัยด้วย bcrypt
  - Role-based access control (Admin, Viewer)

- **🔍 ระบบค้นหา Multi-Target**
  - ค้นหาหลายคนพร้อมกัน
  - ใช้ YOLOv8 สำหรับตรวจจับบุคคล
  - ใช้ ResNet50 สำหรับ Re-ID

- **📸 Feature Extraction**
  - เสกสี dominanat จากเสื้อผ้า
  - ตรวจจับเพศ (Gender Detection)
  - สกัด Embeddings ด้วย ResNet50

- **💾 ฐานข้อมูล**
  - SQLite สำหรับเก็บประวัติ
  - บันทึก Search History
  - บันทึก Target Profiles

- **📧 ระบบแจ้งเตือน**
  - ส่งอีเมลอัตโนมัติ
  - รายงานผลลัพธ์พร้อมสี เพศ

- **🎨 User Interface**
  - Dashboard แบบ Streamlit
  - แท็บ Search และ Result Gallery
  - Sidebar Control Panel

---

## 📊 สรุปความเปลี่ยนแปลง

| เวอร์ชัน | วันที่ | รายละเอียด | ความสำคัญ |
|---------|-------|-----------|---------|
| v2.0.0 | 2026-03-18 | Refactoring + Multilingual | 🔴 Critical |
| v1.5.0 | 2026-03-15 | Admin Panel | 🟠 High |
| v1.0.0 | 2026-03-01 | Initial Release | 🟠 High |

---

## 🎯 แผนพัฒนาในอนาคต (Roadmap)

- [ ] Video streaming support (สตรีมวิดีโอแบบ Real-time)
- [ ] Advanced analytics dashboard
- [ ] API endpoint for 3rd-party integration
- [ ] GPU acceleration support
- [ ] Mobile app
- [ ] Cloud deployment

---

## 👤 ผู้พัฒนา

**Piyaphum Muetkhambong, Boonpithak Phompech**

---
