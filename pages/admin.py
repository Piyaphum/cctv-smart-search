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
import os
import shutil
import datetime
from database import get_history, get_summary_stats
import config
from supabase import create_client, Client

try:
    supabase: Client = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)
except Exception:
    supabase = None

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

# ดึง filter option (User & Date)
c1, c2 = st.columns(2)
with c1:
    filter_user = st.selectbox(
        "กรองตาม User:",
        options=["ทั้งหมด"] + [u for u, _ in stats['top_users']]
    )

with c2:
    # ย้อนหลังไป 7 วันเป็นค่า default
    default_start = datetime.date.today() - datetime.timedelta(days=7)
    date_range = st.date_input("ช่วงเวลา:", [default_start, datetime.date.today()])

# ดึงข้อมูลจาก DB
if filter_user == "ทั้งหมด":
    history = get_history()  # ดึงทั้งหมด (Admin)
else:
    history = get_history(username=filter_user)

if history:
    # แปลงเป็น DataFrame เพื่อแสดงเป็นตาราง
    df = pd.DataFrame(history)
    
    # แปลงคอลัมน์เวลาเป็น datetime เพื่อใช้เทียบ
    df['searched_at'] = pd.to_datetime(df['searched_at'])
    
    # กรองวันที่
    if len(date_range) == 2:
        start_date, end_date = date_range
        # ปรับ end_date ให้คลุมไปถึงสิ้นวัน 23:59:59
        end_date = pd.to_datetime(str(end_date)) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        start_date = pd.to_datetime(str(start_date))
        
        mask = (df['searched_at'] >= start_date) & (df['searched_at'] <= end_date)
        df = df.loc[mask]

    # ฟอร์แมตเวลากลับให้สวยงาม
    df['searched_at'] = df['searched_at'].dt.strftime('%Y-%m-%d %H:%M:%S')

    df = df.rename(columns={
        "id": "ID",
        "username": "User",
        "video_name": "วิดีโอ",
        "target_name": "Target",
        "total_found": "พบ (ครั้ง)",
        "searched_at": "เวลา"
    })
    
    if not df.empty:
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.caption(f"ทั้งหมด {len(df)} รายการ")
    else:
        st.info("ไม่พบข้อมูลในช่วงเวลาที่เลือก")
else:
    st.info("ยังไม่มีประวัติการค้นหา")

st.divider()

# ─────────────────────────────────────────────
# 👥 ระบบจัดการผู้ใช้งาน (User Management)
# ─────────────────────────────────────────────
st.subheader("👥 User Management")
st.markdown("เพิ่มผู้ใช้งานใหม่ รวมถึงดู แก้ไข และลบข้อมูลของระบบทั้งหมดได้จากศูนย์กลางเดียว")

t_add, t_manage = st.tabs(["➕ Add New User", "🛠️ Manage Users"])

with t_add:
    with st.form("new_user_form_admin"):
        st.markdown("**เพิ่มผู้ใช้ใหม่ (Add New User)**")
        new_username = st.text_input("Username")
        
        c1, c2 = st.columns(2)
        with c1: new_fname = st.text_input("First Name (ชื่อจริง)")
        with c2: new_lname = st.text_input("Last Name (นามสกุล)")
        new_name = f"{new_fname.strip()} {new_lname.strip()}".strip()
        
        new_email = st.text_input("Email")
        new_password = st.text_input("Password (min 8 chars)", type="password")
        new_role_label = st.selectbox("Role", options=["viewer", "admin"])
        
        if st.form_submit_button("✅ Create User", type="primary"):
            if not new_username or not new_fname.strip() or not new_lname.strip() or not new_email or not new_password:
                st.error("กรุณากรอกข้อมูลให้ครบถ้วน")
            elif len(new_password) < 8:
                st.error("รหัสผ่านต้องมีความยาวอย่างน้อย 8 ตัวอักษร")
            else:
                try:
                    hashed_pw = stauth.Hasher([new_password]).generate()[0]
                    data = {
                        "username": new_username,
                        "name": new_name,
                        "email": new_email,
                        "password_hash": hashed_pw,
                        "role": new_role_label
                    }
                    if supabase:
                        supabase.table('users').insert(data).execute()
                        st.success("✅ สร้างผู้ใช้ใหม่ลงระบบ Cloud สำเร็จ!")
                        st.rerun()
                    else:
                        st.error("ฐานข้อมูล Supabase ยังไม่พร้อมใช้งาน")
                except Exception as e:
                    if "duplicate" in str(e).lower() or "conflict" in str(e).lower():
                        st.error("แพลตฟอร์มปฏิเสธ: ชื่อผู้ใช้นี้มีอยู่ในระบบแล้ว")
                    else:
                        st.error(f"Error: {e}")

with t_manage:
    st.markdown("ดูและแก้ไขข้อมูลผู้ใช้งานทั้งหมดในระบบ (ดับเบิ้ลคลิกบนช่องในตารางเพื่อแก้ไข จากนั้นกดปุ่ม Save ด้านล่าง)")

    # Load current users from Supabase
    user_data = []
    if supabase:
        try:
            response = supabase.table('users').select('*').execute()
            if response.data:
                for user in response.data:
                    user_data.append({
                        "Username": user["username"],
                        "Name": user["name"],
                        "Email": user["email"],
                        "Role": user.get("role", "viewer")
                    })
        except Exception as e:
            st.error(f"Cannot sync cloud users: {e}")

    df_users = pd.DataFrame(user_data)

    with st.form("user_editor_form_admin"):
        edited_df = st.data_editor(
            df_users,
            hide_index=True,
            use_container_width=True,
            disabled=["Username"], # ล็อค Username ไม่ให้แก้
            column_config={
                "Role": st.column_config.SelectboxColumn(
                    "Role",
                    help="สิทธิ์การใช้งาน",
                    width="medium",
                    options=["admin", "viewer"],
                    required=True
                ),
                "Email": st.column_config.TextColumn(
                    "Email",
                    required=True
                ),
                "Name": st.column_config.TextColumn(
                    "Name",
                    required=True
                )
            }
        )
        
        if st.form_submit_button("💾 Save User Changes", type="primary"):
            try:
                for index, row in edited_df.iterrows():
                    data = {
                        "name": row["Name"],
                        "email": row["Email"],
                        "role": row["Role"]
                    }
                    if supabase:
                        supabase.table('users').update(data).eq('username', row["Username"]).execute()
                    
                st.success("✅ ซิงค์ข้อมูลผู้ใช้งานลง Supabase แล้ว!")
                st.rerun()
            except Exception as e:
                st.error(f"เกิดข้อผิดพลาดในการเชื่อมต่อ Cloud: {e}")

    st.markdown("---")
    st.markdown("**🗑️ Delete User (ลบผู้ใช้)**")
    
    col_del1, col_del2 = st.columns([3, 1])
    with col_del1:
        # ดึงรายชื่อที่เตรียมไว้ก่อนหน้า (df_users)
        available_users = df_users["Username"].tolist() if not df_users.empty else []
        del_user = st.selectbox("เลือกผู้ใช้ออกจากระบบ:", options=[""] + available_users, label_visibility="collapsed")
    with col_del2:
        if st.button("❌ ลบผู้ใช้", use_container_width=True):
            if not del_user:
                st.warning("กรุณาเลือกผู้ใช้ที่ต้องการลบ")
            elif del_user == current_user:
                st.error("❌ ไม่สามารถลบบัญชีที่ตัวเองกำลังใช้งานอยู่ได้")
            else:
                if supabase:
                    supabase.table('users').delete().eq('username', del_user).execute()
                    st.success(f"ลบผู้ใช้ {del_user} ออกจากฐานข้อมูล Cloud สำเร็จ")
                    st.rerun()
                else:
                    st.error("ระบบฐานข้อมูลไม่พร้อมใช้งาน")

st.divider()

# ─────────────────────────────────────────────
# 🧹 เครื่องมือทำความสะอาด (System Clean-up)
# ─────────────────────────────────────────────
st.subheader("🧹 System Clean-up (Admin Only)")
st.markdown("""
**Note:** Admins can clear the entire system cache. 
Regular users can clear their personal cache from the main page.
""")

def get_dir_size(path="."):
    total = 0
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_file(): total += entry.stat().st_size
            elif entry.is_dir(): total += get_dir_size(entry.path)
    return total

RESULT_DIR = "detected_results"
TEMP_DIR = "temp_video"

try:
    res_size = get_dir_size(RESULT_DIR) / (1024*1024) if os.path.exists(RESULT_DIR) else 0
    tmp_size = get_dir_size(TEMP_DIR) / (1024*1024) if os.path.exists(TEMP_DIR) else 0
except:
    res_size, tmp_size = 0, 0

c1, c2 = st.columns(2)
c1.info(f"📁 **Results Folder (All Users):** {res_size:.2f} MB")
c2.info(f"📂 **Temp Videos:** {tmp_size:.2f} MB")

if st.button("🗑️ Clear ALL System Caches", type="primary"):
    # เคลียร์ temp_video
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
        os.makedirs(TEMP_DIR)
    
    # เคลียร์ detected_results
    if os.path.exists(RESULT_DIR):
        shutil.rmtree(RESULT_DIR)
        os.makedirs(RESULT_DIR)
        
    st.success("✅ All system caches cleared successfully!")
    st.rerun()
