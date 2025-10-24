import sys
import logging
from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.email import send_email_smtp

sys.path.append('/opt/airflow/')  # Path to your spark_apps

# Import your modular functions
from spark_apps.preprocess_transform_1 import preprocess_transform
from spark_apps.train_model_1 import train_model
from spark_apps.inference_1 import inference_1
from spark_apps.spark_consumer_1 import kafka_consumer

# ───────────────────────────────
# Configuration
# ───────────────────────────────
MLFLOW_TRACKING_URI = "http://mlflow:5000"
MODEL_URI = "models:/fraud_xgb_model/Production"

INPUT_PATH = "s3a://mlflow-bucket/transactions_features/"
OUTPUT_PATH = "s3a://mlflow-bucket/predictions/"

MINIO_ENDPOINT = "http://minio:9000"
MINIO_ACCESS_KEY = "minio"
MINIO_SECRET_KEY = "minio123"

ALERT_THRESHOLD = 0.05  # alert if predicted fraud ratio > 5%
ALERT_EMAIL = ["jasonling5555@gmail.com"]

# ───────────────────────────────
# Logging setup
# ───────────────────────────────
logger = logging.getLogger("fraud_pipeline")
logger.setLevel(logging.INFO)

# ───────────────────────────────
# DAG utility functions
# ───────────────────────────────
def monitor_predictions(output_path=OUTPUT_PATH, threshold=ALERT_THRESHOLD, email_recipients=ALERT_EMAIL):
    from pyspark.sql import SparkSession
    import pandas as pd

    logger.info("▶ Monitoring fraud predictions...")
    spark = SparkSession.builder.appName("FraudMonitor").getOrCreate()
    spark._jsc.hadoopConfiguration().set("fs.s3a.access.key", MINIO_ACCESS_KEY)
    spark._jsc.hadoopConfiguration().set("fs.s3a.secret.key", MINIO_SECRET_KEY)
    spark._jsc.hadoopConfiguration().set("fs.s3a.endpoint", MINIO_ENDPOINT)
    spark._jsc.hadoopConfiguration().set("fs.s3a.path.style.access", "true")
    spark._jsc.hadoopConfiguration().set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    spark._jsc.hadoopConfiguration().set("fs.s3a.connection.ssl.enabled", "false")

    try:
        df = spark.read.parquet(output_path)
        pdf = df.toPandas()
        fraud_ratio = pdf["predicted_fraud"].mean()
        logger.info(f"Current predicted fraud ratio: {fraud_ratio:.2%}")

        if fraud_ratio > threshold:
            subject = "🚨 Fraud Alert: High Fraud Prediction Ratio"
            body = f"""
            Fraud prediction ratio exceeded threshold!

            Current ratio: {fraud_ratio:.2%}
            Threshold: {threshold:.2%}

            Please investigate recent transactions.
            """
            send_email_smtp(to=email_recipients, subject=subject, html_content=body)
            logger.info("✅ Email alert sent")
        else:
            logger.info("Fraud ratio within acceptable range")
    except Exception as e:
        logger.error(f"Monitoring failed: {e}")
        raise

# ───────────────────────────────
# DAG definition
# ───────────────────────────────
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": True,
    "email": ALERT_EMAIL,
    "retries": 1,  # allow one retry per task
}

with DAG(
    dag_id="fraud_detection_pipeline",
    default_args=default_args,
    description="Simulating Enterprise-level fraud model pipeline (development stage)",
    schedule_interval=None,  # manual trigger only
    start_date=datetime(2025, 10, 1),
    catchup=False,
    tags=["fraud", "mlflow", "minio"],
) as dag:

    # ────────────────
    # Tasks
    # ────────────────
    consumer_task = PythonOperator(
        task_id="kafka_consumer",
        python_callable=kafka_consumer,
    )

    preprocess_task = PythonOperator(
        task_id="preprocess_features",
        python_callable=lambda: preprocess_transform(input_path=INPUT_PATH, output_path=OUTPUT_PATH),
    )

    train_task = PythonOperator(
        task_id="train_model",
        python_callable=lambda: train_model(mlflow_uri=MLFLOW_TRACKING_URI),
    )

    inference_task = PythonOperator(
        task_id="run_inference",
        python_callable=lambda: inference_1(model_uri=MODEL_URI, input_path=OUTPUT_PATH, output_path=OUTPUT_PATH),
    )

    monitor_task = PythonOperator(
        task_id="monitor_predictions",
        python_callable=monitor_predictions,
    )

    # ────────────────
    # Pipeline order
    # ────────────────
    consumer_task >> preprocess_task >> train_task >> inference_task >> monitor_task
