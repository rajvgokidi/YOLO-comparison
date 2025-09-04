from ultralytics import YOLO
from ray import tune
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler
import time
import ray
import os

def yolo_train(config, data_yaml='VOC.yaml'):
    model = YOLO("/home/rgokidi/School/Computer-Vision/runs/detect/train/weights/best.pt")

    results = model.train(
        data=data_yaml,
        epochs=config["epochs"],
        batch=config["batch_size"],
        optimizer="AdamW",
        lr0=config["lr0"],
        lrf=config["lrf"],
        momentum=config["momentum"],
        weight_decay=config["weight_decay"]
    )

    tune.report(map=results.metrics['metrics/mAP50-95(B)'])

search_space = {
    "lr0": tune.uniform(1e-5, 1e-1),
    "lrf": tune.uniform(0.01, 1.0),
    "weight_decay": tune.uniform(0.0, 0.001),
    "momentum": tune.uniform(0.6, 0.98),
    "batch_size": tune.choice([8, 16, 32]),
    "epochs": 30
}

resources_per_trial = {"cpu": 8, "gpu": 1}
trainable_with_resources= tune.with_resources(yolo_train, resources_per_trial)

search_algo = OptunaSearch(metric="metrics/mAP50-95(B)", mode="max")

scheduler = ASHAScheduler(metric="metrics/mAP50-95(B)", mode="max", max_t=30, grace_period=10, reduction_factor=3)

tuner = tune.Tuner(
    trainable_with_resources,
    tune_config=tune.TuneConfig(
        search_alg=search_algo,
        scheduler=scheduler,
        num_samples=30
    ),
    param_space=search_space
)

results = tuner.fit()
best_result = results.get_best_result(metric="metrics/mAP50-95(B)", mode="max")

print("\n--- Best hyperparameters found ---")
print(best_result.config)
print("----------------------------------")
print("Best validation mAP: ", best_result.metrics["metrics/mAP50-95(B)"])

print(f"\nTensorBoard logs are available at: {os.path.join(os.getcwd(), 'ray_results')}")
