import os, re

folder = r'd:\person-reid\detected_results\admin\mix'
files = [f for f in os.listdir(folder) if f.endswith('.jpg')]

report = []
report.append('# ตารางวิเคราะห์จากไฟล์ตรวจจับจริง (Threshold 70%)')
report.append('ตารางด้านล่างเป็นการดึงข้อมูลจริงจากโฟลเดอร์ภาพที่ระบบบันทึกผลลัพธ์ไว้ทั้งหมด 38 ภาพ นำมาจำแนกเรียงตามเวลาในวิดีโอ เพื่อดูว่าถ้าตั้งค่าความแม่นยำ 70% ระบบจะตัดสินใจให้ภาพใดผ่านบ้างครับ\n')

report.append('| ภาพผลลัพธ์จากกล้อง (คลิกเพื่อดูภาพ) | เป้าหมายที่ตามหา (Target) | สีเสื้อตรวจพบ | เวลาในคลิป (Time) | ผลคะแนน (Score) | ตัดสินที่เกณฑ์ 70% |')
report.append('|:---|:---|:---|:---|:---|:---|')

pattern = re.compile(r'Found_(.*?)_(.*?)_(.*?)_([\d.]+)%_(.*?)_(\d+)\.jpg')

data = []
for f in files:
    m = pattern.match(f)
    if m:
        target, color, gender, score_str, time, frame = m.groups()
        score = float(score_str)
        data.append((score, target, color, time, f))

# Sort by time
data.sort(key=lambda x: (x[3], x[1], x[0]))

for score, target, color, time, f in data:
    status = '🚨 **แจังเตือน (Match)**' if score >= 70.0 else '❌ เพิกเฉย'
    file_path = f"file:///{folder.replace(chr(92), '/')}/{f}"
    report.append(f"| [{f}]({file_path}) | {target} | {color} | {time} | **{score}%** | {status} |")

with open('mix_table_real.md', 'w', encoding='utf-8') as fout:
    fout.write('\n'.join(report))
