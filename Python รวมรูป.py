"""
รวมรูปจากโฟลเดอร์ (รวมโฟลเดอร์ย่อย) เป็นรูปตาราง (contact sheet)
แก้พารามิเตอร์ด้านล่างตามต้องการ แล้วรันไฟล์นี้ใน Windows
"""
from pathlib import Path
from PIL import Image
import math
import shutil

# ปรับค่าตามต้องการ
ROOT = Path(r"C:\Users\ASUS\Downloads\7.)Jul-25")  # โฟลเดอร์หลักที่มีรูปและโฟลเดอร์ย่อย
OUT = ROOT / "merged_contact_sheet.jpg"  # ไฟล์ผลลัพธ์
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
IMAGES_PER_ROW = 6       # จำนวนรูปต่อแถว
THUMB_MAX_SIZE = (400, 400)  # ขนาดสูงสุดของแต่ละรูป (width, height)
PADDING = 10             # ช่องว่างระหว่างรูป (px)
BG_COLOR = (30, 30, 30)  # สีพื้นหลัง (R,G,B)

# ปรับค่า
SRC = Path(r"C:\Users\ASUS\Downloads\7.)Jul-25")  # โฟลเดอร์ต้นทาง (มีโฟลเดอร์ย่อย)
DST = Path(r"d:\4.Machine Learning(AI)\Image process\รวมรูปต้นฉบับ_flat2")  # โฟลเดอร์ปลายทาง (จะถูกสร้างถ้ายังไม่มี)
MOVE_INSTEAD_OF_COPY = True  # True = ย้ายไฟล์, False = คัดลอกไฟล์
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

DST.mkdir(parents=True, exist_ok=True)

def unique_target(path: Path) -> Path:
    """คืน Path ที่ไม่ชนชื่อไฟล์ในโฟลเดอร์ปลายทาง (เพิ่ม _1, _2 ... ถ้าชน)"""
    target = DST / path.name
    if not target.exists():
        return target
    stem = path.stem
    suf = path.suffix
    i = 1
    while True:
        candidate = DST / f"{stem}_{i}{suf}"
        if not candidate.exists():
            return candidate
        i += 1

def find_images(root: Path):
    return [p for p in sorted(root.rglob("*")) if p.suffix.lower() in IMAGE_EXTS]

def make_contact_sheet(image_paths, per_row=IMAGES_PER_ROW, thumb_size=THUMB_MAX_SIZE,
                       padding=PADDING, bg_color=BG_COLOR):
    if not image_paths:
        raise RuntimeError("ไม่พบรูปในโฟลเดอร์ที่ระบุ")

    thumbs = []
    max_w = max_h = 0
    for p in image_paths:
        try:
            im = Image.open(p).convert("RGB")
            im.thumbnail(thumb_size, Image.LANCZOS)
            thumbs.append(im)
            max_w = max(max_w, im.width)
            max_h = max(max_h, im.height)
        except Exception as e:
            print(f"ข้าม {p} เนื่องจาก error: {e}")

    cols = per_row
    rows = math.ceil(len(thumbs) / cols)
    cell_w = max_w
    cell_h = max_h

    sheet_w = cols * cell_w + (cols + 1) * padding
    sheet_h = rows * cell_h + (rows + 1) * padding

    sheet = Image.new("RGB", (sheet_w, sheet_h), color=bg_color)

    for idx, im in enumerate(thumbs):
        row = idx // cols
        col = idx % cols
        x = padding + col * (cell_w + padding) + (cell_w - im.width) // 2
        y = padding + row * (cell_h + padding) + (cell_h - im.height) // 2
        sheet.paste(im, (x, y))

    return sheet

moved = 0
skipped = 0
for p in sorted(SRC.rglob("*")):
    if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
        try:
            tgt = unique_target(p)
            if MOVE_INSTEAD_OF_COPY:
                shutil.move(str(p), str(tgt))
            else:
                shutil.copy2(str(p), str(tgt))
            moved += 1
        except Exception as e:
            print(f"ข้าม {p} -> error: {e}")
            skipped += 1

print(f"เสร็จสิ้น: ย้าย/คัดลอก {moved} ไฟล์, ข้าม {skipped} ไฟล์")

if __name__ == "__main__":
    img_paths = find_images(ROOT)
    print(f"พบรูปทั้งหมด: {len(img_paths)} ไฟล์")
    if not img_paths:
        print("ยกเลิก: ไม่มีรูปให้รวม")
    else:
        sheet = make_contact_sheet(img_paths)
        sheet.save(OUT, quality=90)
        print(f"บันทึกไฟล์รวมที่: {OUT}")