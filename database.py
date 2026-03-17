"""
📚 database.py — จัดการ SQLite Database ทั้งหมด
==================================================
ไฟล์นี้ทำหน้าที่เป็น "คลังข้อมูล" ของระบบ
แยกออกมาจาก app.py เพื่อให้โค้ดสะอาด และ ง่ายต่อการแก้ไข

ตาราง (Tables) ที่มี:
  1. users             — เก็บข้อมูลผู้ใช้ (username, role)
  2. search_history    — บันทึกทุกครั้งที่กด Start Search
  3. detections        — บันทึกทุกครั้งที่ตรวจพบบุคคล

ความสัมพันธ์:
  users (1) ──→ (many) search_history (1) ──→ (many) detections
"""

import sqlite3
import datetime

# ชื่อไฟล์ฐานข้อมูล (จะถูกสร้างอัตโนมัติถ้ายังไม่มี)
DB_PATH = "cctv_search.db"


# ─────────────────────────────────────────────
#  SECTION 1: สร้างตาราง (สร้างครั้งเดียวตอน boot)
# ─────────────────────────────────────────────

def init_db():
    """
    สร้างตารางทั้งหมดในฐานข้อมูล
    IF NOT EXISTS = ถ้าตารางมีอยู่แล้ว ไม่ต้องสร้างซ้ำ (ปลอดภัย)
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # ตาราง 1: search_history
    # เก็บว่า user ไหน scan วิดีโออะไร เมื่อไหร่ และเจอกี่ครั้ง
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS search_history (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            username    TEXT    NOT NULL,
            video_name  TEXT    NOT NULL,
            target_name TEXT    NOT NULL,
            total_found INTEGER DEFAULT 0,
            searched_at TEXT    NOT NULL
        )
    """)

    # ตาราง 2: detections
    # เก็บรายละเอียดของแต่ละ detection (เชื่อมกับ search_history ด้วย search_id)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            search_id   INTEGER NOT NULL,
            score       REAL    NOT NULL,
            timestamp_s REAL    NOT NULL,
            FOREIGN KEY (search_id) REFERENCES search_history(id)
        )
    """)

    conn.commit()
    conn.close()
    print("✅ Database initialized:", DB_PATH)


# ─────────────────────────────────────────────
#  SECTION 2: บันทึกข้อมูล (Write)
# ─────────────────────────────────────────────

def log_search(username: str, video_name: str, target_name: str, total_found: int) -> int:
    """
    บันทึกประวัติการค้นหา 1 รายการ
    คืนค่า id ของ record ที่เพิ่งบันทึก (ใช้เชื่อมกับ detections ต่อไป)

    Args:
        username    : ชื่อ user ที่ login อยู่
        video_name  : ชื่อไฟล์วิดีโอที่ scan
        target_name : ชื่อ target ที่ค้นหา
        total_found : จำนวน detection ที่เจอ
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute("""
        INSERT INTO search_history (username, video_name, target_name, total_found, searched_at)
        VALUES (?, ?, ?, ?, ?)
    """, (username, video_name, target_name, total_found, now))
    # หมายเหตุ: ใช้ ? แทน f-string เพื่อป้องกัน SQL Injection!

    search_id = cursor.lastrowid  # ดึง id ของแถวที่เพิ่งเพิ่ม
    conn.commit()
    conn.close()
    return search_id


def log_detection(search_id: int, score: float, timestamp_s: float):
    """
    บันทึก detection 1 ครั้ง (เชื่อมกับ search record)

    Args:
        search_id   : id จาก log_search()
        score       : ค่า confidence score (0.0 - 1.0)
        timestamp_s : เวลาใน video (วินาที)
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO detections (search_id, score, timestamp_s)
        VALUES (?, ?, ?)
    """, (search_id, score, timestamp_s))

    conn.commit()
    conn.close()


# ─────────────────────────────────────────────
#  SECTION 3: ดึงข้อมูล (Read)
# ─────────────────────────────────────────────

def get_history(username: str = None) -> list:
    """
    ดึงประวัติการค้นหา
    ถ้าระบุ username → เฉพาะของ user นั้น
    ถ้าไม่ระบุ (None) → ทั้งหมด (สำหรับ Admin)

    คืนค่าเป็น list ของ dict
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # ทำให้ดึงผลลัพธ์เป็น dict ได้
    cursor = conn.cursor()

    if username:
        cursor.execute("""
            SELECT * FROM search_history
            WHERE username = ?
            ORDER BY searched_at DESC
        """, (username,))
    else:
        cursor.execute("""
            SELECT * FROM search_history
            ORDER BY searched_at DESC
        """)

    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def get_summary_stats() -> dict:
    """
    สรุปสถิติภาพรวม (สำหรับ Admin Dashboard)
    คืนค่า:
        total_searches : จำนวนครั้งที่ scan ทั้งหมด
        total_detected : จำนวน detection ทั้งหมด
        top_users      : list ของ (username, จำนวนครั้งที่ search)
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM search_history")
    total_searches = cursor.fetchone()[0]

    cursor.execute("SELECT SUM(total_found) FROM search_history")
    total_detected = cursor.fetchone()[0] or 0

    cursor.execute("""
        SELECT username, COUNT(*) as cnt
        FROM search_history
        GROUP BY username
        ORDER BY cnt DESC
        LIMIT 5
    """)
    top_users = cursor.fetchall()

    conn.close()
    return {
        "total_searches": total_searches,
        "total_detected": total_detected,
        "top_users": top_users
    }
