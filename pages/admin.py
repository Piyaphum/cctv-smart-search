"""
📊 pages/admin.py — Admin Dashboard
=====================================
Streamlit รองรับ Multi-page App อัตโนมัติ!
ถ้าสร้างโฟลเดอร์ pages/ และวางไฟล์ .py ไว้
มันจะโผล่ใน Sidebar โดยอัตโนมัติ ✨

หน้านี้ทำหน้าที่:
  - แสดงสถิติภาพรวม (total searches, total detections)
  - แสดงประวัติการค้นหาทั้งหมด (Admin เท่านั้น)
"""

import streamlit as st
import yaml
import streamlit_authenticator as stauth
import pandas as pd
from database import get_history, get_summary_stats

st.set_page_config(page_title="Admin Panel | CCTV Search", layout="wide")

# ─────────────────────────────────────────────
# 🔐 ตรวจสอบ Login ก่อน (เหมือนกับ app.py)
# ─────────────────────────────────────────────
with open('auth_config.yaml', 'r', encoding='utf-8') as f:
    auth_config = yaml.safe_load(f)

authenticator = stauth.Authenticate(
    auth_config['credentials'],
    auth_config['cookie']['name'],
    auth_config['cookie']['key'],
    auth_config['cookie']['expiry_days']
)

if not st.session_state.get('authentication_status'):
    st.error("❌ กรุณา Login ที่หน้าหลักก่อน")
    st.stop()  # หยุดทันทีถ้ายังไม่ login

# ────────────────────────────────────
# ดึงข้อมูล user ที่ login อยู่
# ────────────────────────────────────
current_user = st.session_state.get('username', 'unknown')
current_role = auth_config['credentials']['usernames'].get(current_user, {}).get('role', 'viewer')

# ── Sidebar: แสดง user info + Logout ──
with st.sidebar:
    current_name = st.session_state.get('name', 'Unknown')
    st.markdown(f"### 👤 {current_name}")
    st.caption(f"Role: `{current_role}`")
    authenticator.logout('Logout', 'sidebar')

# ─────────────────────────────────────────────
# 🔐 เฉพาะ Admin เท่านั้นที่เข้าหน้านี้ได้
# ─────────────────────────────────────────────
if current_role != 'admin':
    st.error("🚫 หน้านี้สำหรับ Admin เท่านั้น")
    st.info("กลับไปหน้าหลักที่ sidebar ด้านซ้าย")
    st.stop()

# ─────────────────────────────────────────────
# 📊 Admin Dashboard
# ─────────────────────────────────────────────
st.title("🛡️ Admin Dashboard")
st.markdown("ภาพรวมการใช้งานระบบ CCTV Search ทั้งหมด")

# --- สถิติภาพรวม (Metric Cards) ---
stats = get_summary_stats()

col1, col2, col3 = st.columns(3)
col1.metric("📋 Total Searches", stats['total_searches'])
col2.metric("🎯 Total Detections", stats['total_detected'])
col3.metric("👥 Active Users", len(stats['top_users']))

st.divider()

# --- Top Users ---
st.subheader("🏆 Top Users (by search count)")
if stats['top_users']:
    for rank, (uname, count) in enumerate(stats['top_users'], 1):
        st.markdown(f"**{rank}.** `{uname}` — {count} ครั้ง")
else:
    st.info("ยังไม่มีประวัติการค้นหา")

st.divider()

# --- ตารางประวัติทั้งหมด ---
st.subheader("📜 Search History (ทั้งหมด)")

# ดึง filter option
filter_user = st.selectbox(
    "กรองตาม User:",
    options=["ทั้งหมด"] + [u for u, _ in stats['top_users']]
)

# ดึงข้อมูลจาก DB
if filter_user == "ทั้งหมด":
    history = get_history()  # ดึงทั้งหมด (Admin)
else:
    history = get_history(username=filter_user)

if history:
    # แปลงเป็น DataFrame เพื่อแสดงเป็นตาราง
    df = pd.DataFrame(history)
    df = df.rename(columns={
        "id": "ID",
        "username": "User",
        "video_name": "วิดีโอ",
        "target_name": "Target",
        "total_found": "พบ (ครั้ง)",
        "searched_at": "เวลา"
    })
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.caption(f"ทั้งหมด {len(history)} รายการ")
else:
    st.info("ยังไม่มีประวัติการค้นหา")
