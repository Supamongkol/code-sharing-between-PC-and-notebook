import cv2
import json

# --------------------------
# GLOBAL
# --------------------------
drawing = False
ix, iy = -1, -1
rectangles = []
current_label = None
img_display = None


# --------------------------
# Mouse callback
# --------------------------
def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, img_display

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img_copy = img_display.copy()
            cv2.rectangle(img_copy, (ix, iy), (x, y), (0,255,0), 2)
            cv2.imshow("ROI Selector", img_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(img_display, (ix, iy), (x, y), (0,255,0), 2)
        rectangles.append({
            "label": param, 
            "x1": ix, "y1": iy,
            "x2": x,  "y2": y
        })
        cv2.imshow("ROI Selector", img_display)


# --------------------------
# Main function to collect ROIs
# --------------------------
def create_rois(image_path, config_output="config.json"):
    global img_display, rectangles

    img = cv2.imread(image_path)
    img = resize_image_keep_ratio(img)
    img_display = img.copy()

    cv2.namedWindow("ROI Selector")

    # ROI labels: gasket + 8 springs
    roi_labels = ["gasket"] + [f"spring_{i}" for i in range(1, 9)]

    for label in roi_labels:
        print(f"\n👉 Draw ROI for: {label}  (ลากกรอบแล้วปล่อยเมาส์)")
        cv2.setMouseCallback("ROI Selector", draw_rectangle, param=label)

        while True:
            cv2.imshow("ROI Selector", img_display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('n'):   # press 'n' to go to next ROI
                print(f"✔ Saved ROI: {label}")
                break

    cv2.destroyAllWindows()

    # Save all ROIs into config.json
    with open(config_output, "w") as f:
        json.dump(rectangles, f, indent=4)

    print("\n🎉 Saved config.json successfully!")
    return rectangles


# --------------------------
# Helper: Resize function
# --------------------------
def resize_image_keep_ratio(img, width=1280):
    h, w = img.shape[:2]
    scale = width / w
    new_h = int(h * scale)
    return cv2.resize(img, (width, new_h), interpolation=cv2.INTER_AREA)


# --------------------------
# Run example
# --------------------------
if __name__ == "__main__":
    create_rois("input.jpg")
