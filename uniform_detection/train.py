from ultralytics import YOLO

model = YOLO('yolov8n.pt')

model.train(
    data='uniform_detection/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    project='runs/detect',
    name='uniform_parts_detector'
)