Tumor Detection using YOLOv8 and SAM

This project demonstrates tumor detection in brain scan images using *YOLOv8* object detection and *SAM (Segment Anything Model)* for segmentation refinement.

🔧 Requirements

- Python 3.8+
- Jupyter Notebook or any Python IDE
- ultralytics (`pip install ultralytics`)
- torch
- numpy, pandas
- GPU (optional, CPU also supported)

📁 Project Structure


├── data.yaml                  # Dataset configuration file
├── Test_images/              # Folder with test images
├── runs/detect/train/        # YOLOv8 trained weights
├── yolo11n.pt                # YOLOv8 pretrained model
├── sam2_b.pt                 # SAM pretrained model
└── tumor_detection.py        # Main detection script


---

🚀 Training the YOLOv8 Model

python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

train_results = model.train(
    data="C:\\Users\\user\\SAM2_Yolo11\\TumorDetection\\data.yaml",
    epochs=20,
    imgsz=640,
    device='cpu'
)



🔍 Inference on Single Image

python
model = YOLO("runs/detect/train/weights/best.pt")
results = model("Test_images/meningioma_3.png", save=True)
results[0].show()
```

📂 Inference on Image Folder

python
results = model("Test_images", save=True)


📦 Segmentation with SAM

python
from ultralytics import SAM

sam_model = SAM("sam2_b.pt")
results = model("Test_images/meningioma_3.png")

for result in results:
    class_ids = result.boxes.cls.int().tolist()
    if len(class_ids):
        boxes = result.boxes.xyxy
        sam_results = sam_model(result.orig_img, bboxes=boxes, verbose=False, save=True, device='cpu')


✅ Output

- Bounding boxes and masks overlaid on tumor images
- Saved results in runs/ directory

📌 Notes

- Ensure your dataset is YOLO-format and data.yaml is correctly configured.
