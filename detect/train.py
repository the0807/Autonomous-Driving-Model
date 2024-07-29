from ultralytics import YOLO
 
# Load the model
model = YOLO('yolov8x.pt')
 
# Training
results = model.train(
   data = 'dataset/Preprocessed_2DBB/data.yaml',
   imgsz = 640,
   epochs = 250,
   batch = 24,
   device = [0, 1],
   plots = True,
   hsv_h = 0.015,
   hsv_s = 0.7,
   hsv_v = 0.4,
   degrees = 0,
   translate= 0.1,
   scale = 0.5,
   shear = 0,
   perspective = 0,
   flipud = 0,
   fliplr = 0.5,
   bgr = 0,
   mosaic = 1.0,
   mixup = 0,
   copy_paste = 0,
   auto_augment = "randaugment",
   erasing = 0.4,
   crop_fraction = 1.0
)