from ultralytics import YOLO

model = YOLO('./runs/detect/train/weights/best.pt')

validation_results = model.val(
    iou=0.5,
    split='test'
)
