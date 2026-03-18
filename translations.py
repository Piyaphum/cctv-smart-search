"""
Multi-language translations
Support: Thai, English
"""

TRANSLATIONS = {
    "th": {
        # Page titles
        "page_title": "ระบบตรวจจับบุคคล",
        "subtitle": "ค้นหาและระบุตัวตนบุคคลในวิดีโอ CCTV",
        
        # Sidebar
        "system": "ระบบ",
        "user": "ผู้ใช้",
        "language": "ภาษา",
        "logout": "ออกจากระบบ",
        
        # Tabs
        "search": "ค้นหา",
        "results": "ผลลัพธ์",
        
        # Target Setup Section
        "target_setup": "ตั้งค่าเป้าหมาย",
        "saved_profiles": "โปรไฟล์ที่บันทึก",
        "new_upload": "อัปโหลดใหม่",
        "select_profiles": "เลือกโปรไฟล์:",
        "delete": "ลบ",
        "no_saved_profiles": "ยังไม่มีโปรไฟล์ที่บันทึก",
        "upload_images": "อัปโหลดภาพ",
        "save_to_database": "บันทึกลงฐานข้อมูล",
        "save_profile": "บันทึกโปรไฟล์",
        "name_for": "ตั้งชื่อสำหรับ",
        "saved": "บันทึกแล้ว",
        "processing": "กำลังประมวลผล...",
        "targets_selected": "เป้าหมายที่เลือก",
        
        # Email Section
        "email_alerts": "การแจ้งเตือนทางอีเมล",
        "send_alerts": "ส่งการแจ้งเตือนเมื่อพบการจับคู่",
        "recipients": "ผู้รับ",
        
        # Video Search Section
        "video_search": "ค้นหาวิดีโอ",
        "upload_videos": "อัปโหลดไฟล์วิดีโอ",
        "parameters": "พารามิเตอร์",
        "start_search": "เริ่มค้นหา",
        
        # Parameter Names & Help Text
        "similarity": "ความคล้ายคลึง",
        "similarity_help": "ควบคุมความเข้มงวดของการจับคู่ (ค่าสูง = ต้องเหมือนกันเกือบทั้งหมด)",
        
        "color_weight": "น้ำหนักสี",
        "color_weight_help": "วิธีการจับคู่สีเสื้อผ้า (0 = ไม่สนใจสี, 1 = สีต้องเป๊ะ)",
        
        "interval": "ช่วงเวลา (วินาที)",
        "interval_help": "ความถี่ในการนำตัวอย่างเฟรม (น้อยลง = รายละเอียดมากขึ้นแต่ช้า)",
        
        "how_it_works": "วิธีการทำงาน",
        "how_it_works_text": "ระบบจะเปรียบเทียบภาพเป้าหมายกับบุคคลทั้งหมดที่ตรวจพบในเฟรมวิดีโอโดยใช้การจดจำใบหน้า AI",
        
        # Results Section
        "detection_results": "ผลการตรวจจับ",
        "refresh": "รีเฟรช",
        "no_results_yet": "ยังไม่มีผลลัพธ์",
        
        # Documentation
        "documentation": "เอกสาร",
        "quick_start": "เริ่มต้นอย่างรวดเร็ว",
        "add_targets": "เพิ่มเป้าหมาย - อัปโหลดภาพของบุคคลที่ต้องการค้นหา",
        "upload_video": "อัปโหลดวิดีโอ - เลือกวิดีโอ CCTV ที่ต้องการสแกน",
        "configure": "ตั้งค่า - ปรับสลัดเดอร์เพื่อเลือกความคล้ายคลึงและการจับคู่สี",
        "search_process": "ค้นหา - เริ่มกระบวนการตรวจจับ",
        "review_results": "ตรวจสอบ - ดูผลลัพธ์ในแท็บผลลัพธ์",
        
        "tips": "เคล็ดลับ",
        "tip_1": "ใช้ฐานข้อมูลโปรไฟล์เพื่อหลีกเลี่ยงการอัปโหลดซ้ำสำหรับเป้าหมายทั่วไป",
        "tip_2": "ค่าความคล้ายคลึงที่สูงขึ้นช่วยลดผลบวกเท็จ",
        "tip_3": "การแจ้งเตือนอีเมลจะแจ้งให้คุณทราบเมื่อพบการจับคู่",
        
        # Errors
        "error_please_upload": "โปรดอัปโหลดเป้าหมายและวิดีโอ",
        "no_detected_results": "ไม่พบผลลัพธ์",
    },
    
    "en": {
        # Page titles
        "page_title": "Person Detection System",
        "subtitle": "Search and identify individuals across CCTV footage",
        
        # Sidebar
        "system": "System",
        "user": "User",
        "language": "Language",
        "logout": "Logout",
        
        # Tabs
        "search": "Search",
        "results": "Results",
        
        # Target Setup Section
        "target_setup": "Target Setup",
        "saved_profiles": "Saved Profiles",
        "new_upload": "New Upload",
        "select_profiles": "Select profiles:",
        "delete": "Delete",
        "no_saved_profiles": "No saved profiles",
        "upload_images": "Upload images",
        "save_to_database": "Save to database",
        "save_profile": "Save profile",
        "name_for": "Name for",
        "saved": "Saved",
        "processing": "Processing...",
        "targets_selected": "targets selected",
        
        # Email Section
        "email_alerts": "Email Alerts",
        "send_alerts": "Send email alerts when matches found",
        "recipients": "Recipients",
        
        # Video Search Section
        "video_search": "Video Search",
        "upload_videos": "Upload video files",
        "parameters": "Parameters",
        "start_search": "Start Search",
        
        # Parameter Names & Help Text
        "similarity": "Similarity Threshold",
        "similarity_help": "Controls matching strictness (higher = must be almost identical)",
        
        "color_weight": "Color Weight",
        "color_weight_help": "How strictly to match clothing colors (0 = ignore, 1 = must match exactly)",
        
        "interval": "Scan Interval (seconds)",
        "interval_help": "How often to sample frames (lower = more detailed but slower)",
        
        "how_it_works": "How it works",
        "how_it_works_text": "Compares target images against all detected people in video frames using AI facial recognition.",
        
        # Results Section
        "detection_results": "Detection Results",
        "refresh": "Refresh",
        "no_results_yet": "No results yet",
        
        # Documentation
        "documentation": "Documentation",
        "quick_start": "Quick Start",
        "add_targets": "Add targets - Upload photos of people to search for",
        "upload_video": "Upload video - Select CCTV footage to scan",
        "configure": "Configure - Adjust sliders for similarity and color matching",
        "search_process": "Search - Start the detection process",
        "review_results": "Review - Check results in the Results tab",
        
        "tips": "Tips",
        "tip_1": "Use the profiles database to avoid re-uploading common targets",
        "tip_2": "Higher similarity thresholds reduce false positives",
        "tip_3": "Email alerts notify you when matches are found",
        
        # Errors
        "error_please_upload": "Please upload targets and videos",
        "no_detected_results": "No results found",
    }
}


def get_text(key, language="th"):
    """Get translated text"""
    return TRANSLATIONS.get(language, TRANSLATIONS["th"]).get(key, key)
