import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from ultralytics import YOLO
import pandas as pd
import shutil
import cv2
from PIL import Image, ExifTags
import numpy as np
import torch
import threading
from datetime import datetime

# ---------------- Folder selectors ----------------
def select_image_folder():
    folder_selected = filedialog.askdirectory()
    image_folder_var.set(folder_selected)

# ---------------- Orientation fix ----------------
def fix_orientation(image_path, image, preferred="horizontal", mode="auto"):
    try:
        pil_img = Image.open(image_path)
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = pil_img._getexif()
        if exif:
            val = exif.get(orientation, 1)
            if val == 3:
                image = cv2.rotate(image, cv2.ROTATE_180)
            elif val == 6:
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            elif val == 8:
                image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    except:
        pass
    return image

# ---------------- 4-angle prediction ----------------
def predict_best_orientation(model, img_path, conf_threshold, debug_dir, save_debug=False):
    image = cv2.imread(img_path)
    if image is None:
        raise ValueError(f"ไม่สามารถอ่านรูปภาพได้: {img_path}")

    rotations = {
        0: image,
        90: cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE),
        180: cv2.rotate(image, cv2.ROTATE_180),
        270: cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    }

    best_result = None
    best_angle = 0
    best_count = -1
    best_conf_sum = -1.0

    os.makedirs(debug_dir, exist_ok=True)

    for angle, rotated_img in rotations.items():
        results = model.predict(source=rotated_img, conf=conf_threshold, verbose=False)
        result = results[0]
        count = len(result.boxes)
        conf_sum = float(result.boxes.conf.sum()) if count > 0 else 0.0

        if save_debug:
            debug_out = os.path.join(debug_dir, f"{os.path.basename(img_path)}_{angle}deg.jpg")
            result.save(filename=debug_out)

        if (count > best_count) or (count == best_count and conf_sum > best_conf_sum):
            best_result = result
            best_angle = angle
            best_count = count
            best_conf_sum = conf_sum

    return best_result, best_angle, best_count, best_conf_sum

# ---------------- Loader with progress ----------------
def show_loader_with_progress(root, total_images):
    loader = tk.Toplevel(root)
    loader.title("กำลังประมวลผล...")
    loader.geometry("400x130")
    loader.resizable(False, False)
    tk.Label(loader, text="กำลังประมวลผลภาพ...").pack(pady=10)

    progress_var = tk.DoubleVar()
    progress = ttk.Progressbar(loader, variable=progress_var, maximum=total_images)
    progress.pack(pady=10, fill="x", padx=20)

    percent_label = tk.Label(loader, text="0 / {}".format(total_images))
    percent_label.pack()

    loader.update()
    return loader, progress_var, progress, percent_label

def update_loader(loader, progress_var, progress, percent_label, current):
    percent = (current / progress["maximum"]) * 100
    progress_var.set(current)
    percent_label.config(text=f"{percent:.0f}% ({int(current)} / {int(progress['maximum'])})")
    loader.update()

# ---------------- Main prediction ----------------
def run_prediction_threaded():
    image_folder = image_folder_var.get()
    if not image_folder:
        messagebox.showerror("ข้อผิดพลาด", "กรุณาเลือกโฟลเดอร์รูปภาพก่อน")
        return
    all_images = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    if not all_images:
        messagebox.showerror("ข้อผิดพลาด", "ไม่พบรูปภาพในโฟลเดอร์นี้")
        return

    loader, progress_var, progress, percent_label = show_loader_with_progress(root, len(all_images))

    def task():
        try:
            run_prediction(loader, progress_var, progress, percent_label)
        finally:
            loader.destroy()

    threading.Thread(target=task, daemon=True).start()

def run_prediction(loader=None, progress_var=None, progress=None, percent_label=None):
    image_folder = image_folder_var.get()
    conf_threshold = conf_slider.get()
    resize_size = int(resize_var.get())
    rotation_mode = rotation_mode_var.get()
    save_debug = save_debug_var.get()

    current_dir = os.getcwd()
    model_path = os.path.join(current_dir, "models", "best.pt")
    if not os.path.exists(model_path):
        messagebox.showerror("ข้อผิดพลาด", "ไม่พบไฟล์ models/best.pt ในโฟลเดอร์โปรแกรม")
        return

    # ✅ สร้างโฟลเดอร์ output พร้อมวันเวลา
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_folder = os.path.join(current_dir, f"output_{timestamp}")
    os.makedirs(output_folder, exist_ok=True)
    waiting_folder = os.path.join(output_folder, "Waiting for verification")
    os.makedirs(waiting_folder, exist_ok=True)
    debug_root = os.path.join(output_folder, "debug_rotations")
    os.makedirs(debug_root, exist_ok=True)

    model = YOLO(model_path)
    excel_data = []
    all_images = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for idx, img_name in enumerate(all_images, start=1):
        img_path = os.path.join(image_folder, img_name)
        image = cv2.imread(img_path)
        corrected = fix_orientation(img_path, image, mode=rotation_mode)

        debug_dir = os.path.join(debug_root, os.path.splitext(img_name)[0])
        best_result, best_angle, det_count, conf_sum = predict_best_orientation(
            model, img_path, conf_threshold, debug_dir, save_debug
        )

        output_image_path = os.path.join(output_folder, f"output_{img_name}")
        best_result.save(filename=output_image_path)

        labels = best_result.names
        detections = best_result.boxes.cls.tolist()
        predicted_labels = [labels[int(d)] for d in detections] if detections else ["No detection"]

        num_correct = predicted_labels.count("correct")
        has_other_class = any(lbl != "correct" for lbl in predicted_labels)

        if num_correct != 8 or has_other_class:
            dst_path = os.path.join(waiting_folder, f"output_{img_name}")
            shutil.move(output_image_path, dst_path)

        excel_data.append({
            "ชื่อภาพ": img_name,
            "มุมที่ใช้": f"{best_angle}°",
            "จำนวน Detection": det_count,
            "Confidence รวม": round(conf_sum, 3),
            "ผลการทำนาย": ", ".join(predicted_labels),
            "สถานะ": "ต้องตรวจสอบ" if num_correct != 8 or has_other_class else "ปกติ"
        })

        if loader:
            update_loader(loader, progress_var, progress, percent_label, idx)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    report_path = os.path.join(output_folder, "Prediction_Results.xlsx")
    df = pd.DataFrame(excel_data)
    df.to_excel(report_path, index=False)
    messagebox.showinfo("สำเร็จ", f"บันทึกผลลัพธ์ที่:\n{output_folder}\n\nและไฟล์ Excel: {report_path}")

# ---------------- GUI ----------------
root = tk.Tk()
root.title("AI Prediction GUI")

image_folder_var = tk.StringVar()
rotation_mode_var = tk.StringVar(value="auto")
save_debug_var = tk.BooleanVar(value=False)

tk.Label(root, text="เลือกโฟลเดอร์รูปภาพ:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
tk.Entry(root, textvariable=image_folder_var, width=50).grid(row=0, column=1)
tk.Button(root, text="เลือก", command=select_image_folder).grid(row=0, column=2)

tk.Label(root, text="Confidence Threshold:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
conf_slider = tk.Scale(root, from_=0.1, to=1.0, orient="horizontal", resolution=0.05, length=300)
conf_slider.set(0.7)
conf_slider.grid(row=1, column=1, sticky="w")

# ✅ Resize selectable only in common values
tk.Label(root, text="ขนาด Resize:").grid(row=2, column=0, padx=10, pady=5, sticky="w")
resize_values = [240, 360, 640, 720, 1280, 1920]
resize_var = tk.IntVar(value=640)
resize_combo = ttk.Combobox(root, textvariable=resize_var, values=resize_values, width=10, state="readonly")
resize_combo.grid(row=2, column=1, sticky="w")

tk.Label(root, text="โหมดการหมุนภาพ:").grid(row=3, column=0, padx=10, pady=5, sticky="w")
tk.Radiobutton(root, text="ไม่หมุน", variable=rotation_mode_var, value="none").grid(row=3, column=1, sticky="w")
tk.Radiobutton(root, text="บังคับแนวนอน", variable=rotation_mode_var, value="force").grid(row=4, column=1, sticky="w")
tk.Radiobutton(root, text="Auto (วิเคราะห์แนวชิ้นงาน)", variable=rotation_mode_var, value="auto").grid(row=5, column=1, sticky="w")

tk.Checkbutton(root, text="บันทึกรูป Debug (ช้าลงเล็กน้อย)", variable=save_debug_var).grid(row=6, column=1, sticky="w", pady=5)

tk.Button(root, text="รันการทำนาย", command=run_prediction_threaded, bg="green", fg="white").grid(row=7, column=1, pady=20)

root.mainloop()
