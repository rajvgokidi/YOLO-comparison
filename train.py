import comet_ml
from ultralytics import YOLO

comet_ml.login()

model = YOLO("yolo11n.yaml")

results = model.train(data="VOC.yaml", epochs=100, lr0=0.005119, lrf=0.6209, weight_decay=0.0001926, momentum=0.7649, optimizer="AdamW",imgsz=640)
