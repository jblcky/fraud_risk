from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.python import PythonOperator
from pyspark.sql import SparkSession
import mlflow
import boto3
from botocore.exceptions import ClientError

default_args = {'owner': 'airflow', 'retries': 1}

dag = DAG(
    'health_check_dag',
    default_args=default_args,
    description='Health check for Spark + MinIO + MLflow',
    schedule_interval=None,
    start_date=days_ago(1),
    catchup=False,
)

# --- 1. Spark health check ---
def spark_check():
    spark = SparkSession.builder \
        .master("spark://spark-master:7077") \
        .appName("health_check_spark") \
        .getOrCreate()

    # Simple job: count numbers from 1 to 10
    data = spark.range(1, 11)
    print(f"Count result: {data.count()}")
    spark.stop()

spark_test = PythonOperator(
    task_id='spark_test',
    python_callable=spark_check,
    dag=dag
)

# --- 2. MinIO health check ---
def minio_check():
    s3 = boto3.client(
        's3',
        endpoint_url='http://minio:9000',
        aws_access_key_id='minio',
        aws_secret_access_key='minio123'
    )
    try:
        s3.put_object(Bucket='mlflow-bucket', Key='health_check.txt', Body=b'OK')
        print("MinIO write OK")
    except ClientError as e:
        raise Exception(f"MinIO test failed: {e}")

minio_test = PythonOperator(
    task_id='minio_test',
    python_callable=minio_check,
    dag=dag
)

# --- 3. MLflow health check ---
def mlflow_check():
    mlflow.set_tracking_uri("postgresql+psycopg2://mlflow:changeme@mlflow-db:5432/mlflow_db")
    mlflow.set_experiment("health_check_experiment")
    with mlflow.start_run(run_name="health_check_run") as run:
        mlflow.log_metric("test_metric", 1.0)
    print("MLflow log OK")

mlflow_test = PythonOperator(
    task_id='mlflow_test',
    python_callable=mlflow_check,
    dag=dag
)

# --- 4. MLflow + MinIO artifact log check ---
def mlflow_minio_check():
    import mlflow
    import os

    # Set tracking URI to Postgres
    mlflow.set_tracking_uri("postgresql+psycopg2://mlflow:changeme@mlflow-db:5432/mlflow_db")

    # Set experiment
    mlflow.set_experiment("health_check_experiment")

    # Start run
    with mlflow.start_run(run_name="health_check_run") as run:
        # Log a metric
        mlflow.log_metric("test_metric", 1.0)

        # Create a dummy artifact file
        artifact_path = "/tmp/health_check.txt"
        with open(artifact_path, "w") as f:
            f.write("Hello MLflow + MinIO!")

        # Log artifact to MLflow (S3/MinIO)
        mlflow.log_artifact(artifact_path)

    print("MLflow metric + artifact log OK")


mlflow_artifact_test = PythonOperator(
    task_id='mlflow_minio_test',
    python_callable=mlflow_minio_check,
    dag=dag
)


def spark_minio_write_test():
    from pyspark.sql import SparkSession

    jar_path = "/opt/spark/jars/hadoop-aws-3.3.4.jar,/opt/spark/jars/aws-java-sdk-bundle-1.12.262.jar"

    spark = SparkSession.builder \
        .appName("health_check_spark_minio") \
        .config("spark.jars", jar_path) \
        .config("spark.driver.extraClassPath", jar_path) \
        .config("spark.executor.extraClassPath", jar_path) \
        .getOrCreate()

    # Set MinIO credentials (S3A)
    hadoop_conf = spark._jsc.hadoopConfiguration()
    hadoop_conf.set("fs.s3a.endpoint", "http://minio:9000")
    hadoop_conf.set("fs.s3a.access.key", "minio")
    hadoop_conf.set("fs.s3a.secret.key", "minio123")
    hadoop_conf.set("fs.s3a.path.style.access", "true")
    hadoop_conf.set("fs.s3a.connection.ssl.enabled", "false")
    hadoop_conf.set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")

    # Create a simple DataFrame
    data = [("Alice", 1), ("Bob", 2), ("Charlie", 3)]
    df = spark.createDataFrame(data, ["name", "id"])

    # Write DataFrame to MinIO as Parquet
    df.write.mode("overwrite").parquet("s3a://mlflow-bucket/health_check_spark.parquet")

    print("Spark → MinIO write OK")


spark_minio_test = PythonOperator(
    task_id='spark_minio_test',
    python_callable=spark_minio_write_test,
    dag=dag
)

# --- Dependencies ---
spark_test >> minio_test >> mlflow_test >> spark_minio_test >> mlflow_artifact_test
