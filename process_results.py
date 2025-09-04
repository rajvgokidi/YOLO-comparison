from ray.tune import ExperimentAnalysis

# Load the experiment results
analysis = ExperimentAnalysis("/home/rgokidi/School/Computer-Vision/runs/detect/tune3")

# Get all results as a dataframe
df = analysis.dataframe()
print("Available metrics:", df.columns.tolist())

# Get the best trial (replace 'your_metric' with actual metric name)
best_result = analysis.get_best_config(metric="metrics/mAP50-95(B)", mode="max")  # or "min"
print(f"Best config: {best_result}")

# Get best trial object
best_trial = analysis.get_best_trial(metric="metrics/mAP50-95(B)", mode="max")
print(f"Best trial results: {best_trial.last_result}")
