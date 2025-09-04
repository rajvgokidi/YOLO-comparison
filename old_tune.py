from ultralytics import YOLO
from ray import tune

model = YOLO("/home/rgokidi/School/Computer-Vision/runs/detect/train3/weights/best.pt")

search_space = {
    "lr0": tune.uniform(1e-5, 1e-1),
    "lrf": tune.uniform(0.01, 1.0),
    "weight_decay": tune.uniform(0.0, 0.001),
    "momentum": tune.uniform(0.6, 0.98),
}

result_grid = model.tune(
    data="VOC.yaml",
    space = search_space,
    epochs = 50,
    gpu_per_trial = 1,
    use_ray = True
)
