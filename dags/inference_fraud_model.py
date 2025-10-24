from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import sys
import os
import mlflow
import pandas as pd
from pyspark.sql import SparkSession

sys.path.append("/opt/airflow")

from utils.logger import get_logger
from utils.send_notifications import send_notification
from spark_apps.inference_1 import inference_1
from spark_apps.spark_consumer_1 import kafka_consumer
from utils.spark_utils import get_spark_session

logger = get_logger(__name__)

# -----------------------------
# Config
# -----------------------------
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_URI = os.getenv("MODEL_URI", "models:/fraud_xgb_model/Production")
INPUT_PATH = os.getenv("INPUT_PATH", "s3a://mlflow-bucket/transactions_features/")
OUTPUT_PATH = os.getenv("OUTPUT_PATH", "s3a://mlflow-bucket/predictions/")
ALERT_THRESHOLD = float(os.getenv("ALERT_THRESHOLD", 0.05))  # 5%

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": True,
    "email": ['jasonling5555@gmail.com'],
    "retries": 1,
}

# -----------------------------
# DAG Definition
# -----------------------------
with DAG(
    dag_id="fraud_inference_pipeline",
    default_args=default_args,
    description="Run fraud model inference and monitor predictions",
    schedule_interval=None,  # manual or event-based trigger
    start_date=datetime(2025, 10, 1),
    catchup=False,
    tags=["fraud", "mlflow", "inference"],
) as dag:

    # consumer_task = PythonOperator(
    #     task_id="kafka_consumer",
    #     python_callable=kafka_consumer,
    # )

    inference_task = PythonOperator(
        task_id="run_inference",
        python_callable=inference_1,
        op_kwargs={
            "input_path": INPUT_PATH,
            "output_path": OUTPUT_PATH,
            "model_uri": MODEL_URI
        },
    )

    def monitor_predictions():
        spark = get_spark_session("FraudPredictionMonitor")
        df = spark.read.parquet(OUTPUT_PATH)
        fraud_ratio = df.toPandas()["predicted_fraud"].mean()
        logger.info("Current predicted fraud ratio: %.2f%%", fraud_ratio * 100)

        if fraud_ratio > ALERT_THRESHOLD:
            send_notification(
                subject="🚨 Fraud Alert: High Prediction Ratio",
                message=f"Predicted fraud ratio {fraud_ratio:.2%} exceeded threshold {ALERT_THRESHOLD:.2%}"
            )

    monitor_task = PythonOperator(
        task_id="monitor_predictions",
        python_callable=monitor_predictions,
    )

    # Pipeline order
    inference_task >> monitor_task
