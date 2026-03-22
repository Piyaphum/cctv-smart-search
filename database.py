"""
📚 database.py — จัดการ Supabase Cloud Database ทั้งหมด
==================================================
ไฟล์นี้ทำหน้าที่เป็น "คลังข้อมูล" ของระบบบน Cloud
"""

from supabase import create_client, Client
import config
import numpy as np

try:
    supabase: Client = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)
except Exception as e:
    print(f"Failed to initialize supabase client: {e}")
    supabase = None

# ฟังก์ชั่นช่วยแปลง numpy array ไปเป็น list สำหรับ json
def convert_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list) or isinstance(obj, tuple):
        return [convert_to_list(i) for i in obj]
    # ในกรณีที่เป็น single float (numpy float32, etc.)
    elif hasattr(obj, 'item'): 
        return obj.item()
    return obj

# ─────────────────────────────────────────────
#  SECTION 1: บันทึกข้อมูล (Write)
# ─────────────────────────────────────────────

def log_search(username: str, video_name: str, target_name: str, total_found: int) -> str:
    """บันทึกประวัติการค้นหา 1 รายการ คืนค่า uuid ของ record"""
    if supabase is None: return None
    data = {
        "username": username,
        "video_name": video_name,
        "target_name": target_name,
        "total_found": total_found
    }
    response = supabase.table("search_history").insert(data).execute()
    if response.data:
        return response.data[0]["id"]
    return None

def log_detection(search_id: str, score: float, timestamp_s: float):
    """บันทึก detection 1 ครั้ง (เชื่อมกับ search record ด้วย UUID)"""
    if supabase is None or not search_id: return
    data = {
        "search_id": search_id,
        "score": float(score),
        "timestamp_s": float(timestamp_s)
    }
    supabase.table("detections").insert(data).execute()

def save_target_profile(name: str, embeddings: list, hists_full: list, hists_top: list, created_by: str) -> str:
    """บันทึก Target ที่ประมวลผล AI เสร็จแล้วลง Cloud Database"""
    if supabase is None: return None
    data = {
        "name": name,
        "embeddings": convert_to_list(embeddings),
        "hists_full": convert_to_list(hists_full),
        "hists_top": convert_to_list(hists_top),
        "created_by": created_by
    }
    response = supabase.table("target_profiles").insert(data).execute()
    if response.data:
        return response.data[0]["id"]
    return None

# ─────────────────────────────────────────────
#  SECTION 2: ดึงข้อมูล (Read)
# ─────────────────────────────────────────────

def get_history(username: str = None) -> list:
    """ดึงประวัติการค้นหา ทั้งหมด หรือแยกตามผู้ใช้"""
    if supabase is None: return []
    try:
        if username:
            response = supabase.table("search_history").select("*").eq("username", username).order("searched_at", desc=True).execute()
        else:
            response = supabase.table("search_history").select("*").order("searched_at", desc=True).execute()
        return response.data
    except:
        return []

def get_summary_stats() -> dict:
    """สรุปสถิติภาพรวม (สำหรับ Admin Dashboard)"""
    if supabase is None:
        return {"total_searches": 0, "total_detected": 0, "top_users": []}
        
    try:
        response = supabase.table("search_history").select("username, total_found").execute()
        rows = response.data
        
        total_searches = len(rows)
        total_detected = sum(r.get("total_found", 0) for r in rows)
        
        user_counts = {}
        for r in rows:
            uname = r.get("username", "unknown")
            user_counts[uname] = user_counts.get(uname, 0) + 1
            
        sorted_users = sorted(user_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "total_searches": total_searches,
            "total_detected": total_detected,
            "top_users": sorted_users
        }
    except:
        return {"total_searches": 0, "total_detected": 0, "top_users": []}

def get_all_target_profiles() -> list:
    """โหลด Target Profiles ทั้งหมดที่บันทึกไว้ และทำการแปลง List กลับมาเป็น Numpy array เพื่อให้โมเดลคำนวณได้"""
    if supabase is None: return []
    try:
        response = supabase.table("target_profiles").select("id, name, embeddings, hists_full, hists_top, created_by, created_at").order("created_at", desc=True).execute()
        
        profiles = []
        for r in response.data:
            profiles.append({
                "id": r["id"],
                "name": r["name"],
                "embeddings": [np.array(x) for x in r["embeddings"]] if r["embeddings"] else [],
                "hists_full": [np.array(x) for x in r["hists_full"]] if r["hists_full"] else [],
                "hists_top": [np.array(x) for x in r["hists_top"]] if r["hists_top"] else [],
                "created_by": r.get("created_by", "unknown"),
                "created_at": r.get("created_at", ""),
                "image": None
            })
        return profiles
    except Exception as e:
        print(f"Error loading targets: {e}")
        return []

def delete_target_profile(profile_id: str):
    """ลบ Profile"""
    if supabase is None: return
    try:
        supabase.table("target_profiles").delete().eq("id", profile_id).execute()
    except Exception as e:
        pass

def init_db():
    print("✅ System adapted for Supabase Cloud Database")
