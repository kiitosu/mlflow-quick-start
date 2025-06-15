import mlflow

# Creates local mlruns directory for experiments
mlflow.set_tracking_uri("http://localhost:5001")
mlflow.set_experiment("my-genai-experiment")
