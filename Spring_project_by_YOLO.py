from ultralytics import YOLO


def main():
    # โหลดโมเดล YOLO ที่ pre-trained
    model = YOLO("yolov8n.pt")  # หรือใช้โมเดลอื่น เช่น yolov8s.pt

    # เริ่มการเทรน
    model.train(
        data="gasket_spring_detection.yaml",
        epochs=50,
        imgsz=640,
        batch=2,
        name="gasket_spring_model_sep2025",
        workers=2,
        augment=True
    )
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()  # 👈 ป้องกัน error multiprocessing บน Windows
    main()