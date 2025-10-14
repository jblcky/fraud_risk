import mlflow
import os

try:
    # Connect to MLflow
    mlflow.set_tracking_uri("http://mlflow:5000")
    print("✅ Connected to MLflow at http://mlflow:5000")

    # List experiments
    experiments = mlflow.search_experiments()
    print(f"Found {len(experiments)} experiments:")
    for exp in experiments:
        print(f"  - {exp.name} (ID: {exp.experiment_id})")

    # Get runs from "FraudDetection" experiment
    client = mlflow.tracking.MlflowClient()
    runs = client.search_runs("FraudDetection")

    if not runs:
        print("⚠️  No runs found in 'FraudDetection' experiment")
    else:
        print(f"\nFound {len(runs)} runs in 'FraudDetection':")
        for run in runs:
            print(f"Run ID: {run.info.run_id}")
            print(f"Status: {run.info.status}")
            print(f"Start Time: {run.info.start_time}")
            print("-" * 40)

except Exception as e:
    print(f"❌ Error: {e}")
    print("Check if MLflow is running: http://localhost:5000")
