#Keyenece Concept
# Screw_FG_appearance_inspection
import cv2
import numpy as np

# ========== CONFIG ==========
TEMPLATE_PATH = "screw_FG_app.jpg"
INPUT_PATH = "input.jpg"

MATCH_THRESHOLD = 0.78            # ถ้าต่ำกว่า = gasket หาย / วางผิด
SPRING_MIN_EDGE = 40               # ตรวจว่ามีสปริงใน ROI หรือไม่
SHOW_DEBUG = True
# =============================


# โหลดรูป
template = cv2.imread(TEMPLATE_PATH, 0)
img = cv2.imread(INPUT_PATH)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ---- A) DETECT GASKET BY TEMPLATE MATCHING ----
result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

h, w = template.shape

if SHOW_DEBUG:
    print("Gasket match value:", max_val)

if max_val < MATCH_THRESHOLD:
    print("❌ GASKET NG! ไม่พบ gasket หรือวางผิดตำแหน่ง")
    exit()

# วงสี่เหลี่ยมตำแหน่ง gasket
top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)
cv2.rectangle(img, top_left, bottom_right, (0,255,0), 2)
print("✔ GASKET OK")


# ---- B) GENERATE 8 SPRING ROIs ----
# ROI ถูกคำนวณแบบอัตโนมัติ โดยอิงจากตำแหน่ง gasket template
roi_list = []
roi_size = 40                     # ขนาด ROI รอบรูสปริง

# ตำแหน่งรูวงกลมบน gasket (normalized)
# ค่าเหล่านี้ผมตั้งจากโครงสร้าง gasket ในรูปจริง
circle_offsets = [
    (0.18, 0.20),
    (0.18, 0.50),
    (0.18, 0.80),

    (0.50, 0.20),
    (0.50, 0.80),

    (0.82, 0.20),
    (0.82, 0.50),
    (0.82, 0.80)
]

for ox, oy in circle_offsets:
    cx = int(top_left[0] + w * ox)
    cy = int(top_left[1] + h * oy)
    roi_list.append((cx, cy))

# ---- C) ตรวจสปริงในแต่ละ ROI ----
spring_status = []

for i, (cx, cy) in enumerate(roi_list):
    x1, y1 = cx - roi_size, cy - roi_size
    x2, y2 = cx + roi_size, cy + roi_size

    roi = gray[y1:y2, x1:x2]

    # ตรวจว่ามี spring ด้วย edge detection
    edges = cv2.Canny(roi, 80, 160)
    edge_count = np.sum(edges > 0)

    if SHOW_DEBUG:
        print(f"ROI {i+1}: edge={edge_count}")

    if edge_count < SPRING_MIN_EDGE:
        spring_status.append("MISSING")
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), 2)
        cv2.putText(img, "NG", (cx-10, cy), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0,0,255), 2)
    else:
        spring_status.append("OK")
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)

# ---- SUMMARY ----
print("\n=== SPRING CHECK RESULT ===")
for i, s in enumerate(spring_status, 1):
    print(f"Spring {i}: {s}")

if all(s == "OK" for s in spring_status):
    print("\n✔ FINAL RESULT: ALL OK")
else:
    print("\n❌ FINAL RESULT: NG")

# แสดงผล
cv2.imshow("Inspection Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
