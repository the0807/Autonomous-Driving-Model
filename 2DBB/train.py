from ultralytics import YOLO
 
# Load the model
model = YOLO('yolov8l.pt')
 
# Training
results = model.train(
   data='./dataset/Preprocessed_2DBB/data.yaml',
   imgsz=640,
   epochs=300,
   batch=24,
   device=0,
   plots=True,
)