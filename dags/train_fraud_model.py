from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import sys
import os

# Add airflow container base path
sys.path.append("/opt/airflow/")

from utils.logger import get_logger
from utils.send_notifications import send_notification
from spark_apps.preprocess_transform_1 import preprocess_transform
from spark_apps.train_model_1 import train_model

logger = get_logger(__name__)

# -----------------------------
# Config
# -----------------------------
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
INPUT_PATH = os.getenv("INPUT_PATH", "s3a://mlflow-bucket/transactions_features/")
OUTPUT_PATH = os.getenv("OUTPUT_PATH", "s3a://mlflow-bucket/model_ready/")

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": True,
    "email": ["jasonling5555@gmail.com"],
    "retries": 1,
}

# -----------------------------
# DAG Definition
# -----------------------------
with DAG(
    dag_id="fraud_training_pipeline",
    default_args=default_args,
    description="Preprocess and train fraud model (training DAG)",
    schedule_interval=None,  # manual trigger or external trigger
    start_date=datetime(2025, 10, 1),
    catchup=False,
    tags=["fraud", "mlflow", "training"],
) as dag:

    preprocess_task = PythonOperator(
        task_id="preprocess_features",
        python_callable=preprocess_transform,
        op_kwargs={"input_path": INPUT_PATH, "output_path": OUTPUT_PATH},
    )

    train_task = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
        op_kwargs={"input_path": OUTPUT_PATH, "mlflow_uri": MLFLOW_TRACKING_URI},
    )

    # Optional notification after training
    def notify_success():
        send_notification(
            subject="✅ Fraud Model Training Completed",
            message="The fraud model training DAG completed successfully."
        )

    notify_task = PythonOperator(
        task_id="notify_training_success",
        python_callable=notify_success,
    )

    # Pipeline order
    preprocess_task >> train_task >> notify_task
