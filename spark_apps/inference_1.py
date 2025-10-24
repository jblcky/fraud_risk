def inference_1(model_uri: str, input_path: str, output_path: str):
    """
    Enterprise-style development-friendly Spark batch inference for fraud detection.
    - Loads latest Production model from MLflow Model Registry
    - Reads feature parquet from MinIO
    - Validates schema
    - Runs prediction (probability + binary fraud label)
    - Writes minimal output back to MinIO
    - Logs run info to MLflow
    """

    import time
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import col, lit, udf
    from pyspark.sql.types import DoubleType, IntegerType
    import mlflow.pyfunc
    import mlflow
    import os
    from pyspark.sql.functions import col, hour, dayofweek
    from datetime import datetime
    import logging
    from utils.spark_utils import get_spark_session

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # -------------------------
    # 1. CONFIGURATION
    # -------------------------
    # MLflow / MinIO environment
    os.environ["MLFLOW_TRACKING_URI"] = "http://mlflow:5000"
    os.environ["AWS_ACCESS_KEY_ID"] = "minio"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://minio:9000"

    # Load the latest Production model
    # MODEL_URI = "models:/fraud_xgb_model/Production"

    # # Input/Output paths (MinIO s3a)
    # INPUT_PATH = "s3a://mlflow-bucket/transactions_features/"
    # OUTPUT_PATH = "s3a://mlflow-bucket/predictions/"

    # Features used during training
    FEATURE_COLS = [
        "is_high_value",
        "is_foreign_tx",
        "is_rapid_tx",
        "is_new_merchant",
        "is_amount_anomaly",
        "hour_of_day",
        "day_of_week",
        "avg_amount_user",
        "std_amount_user"
    ]

    # Output columns
    PROB_COL = "fraud_probability"
    PRED_COL = "predicted_fraud"
    TIMESTAMP_COL = "inference_timestamp"

    # -------------------------
    # 2. Spark session
    # -------------------------
    spark = get_spark_session("FraudBatchInference")

    # -------------------------
    # 3. Load MLflow Production model
    # -------------------------
    logger.info(f"Loading latest Production model from MLflow: {model_uri}")
    model = mlflow.pyfunc.load_model(model_uri)

    # -------------------------
    # 4. Read input parquet
    # -------------------------
    logger.info(f"Reading input data from: {input_path}")
    df = spark.read.parquet(input_path)

    df = df.withColumn("hour_of_day", hour(col("event_time")))
    df = df.withColumn("day_of_week", dayofweek(col("event_time")))

    # -------------------------
    # 5. Validate feature columns
    # -------------------------
    missing = [f for f in FEATURE_COLS if f not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")
    if "transaction_id" not in df.columns:
        raise ValueError("Missing required column: transaction_id")

    # -------------------------
    # 6. Define prediction UDF
    # -------------------------
    predict_udf = udf(lambda *cols: float(model.predict([cols])[0]), DoubleType())

    df_pred = df.withColumn(PROB_COL, predict_udf(*[col(f) for f in FEATURE_COLS]))
    df_pred = df_pred.withColumn(PRED_COL, (col(PROB_COL) >= 0.98).cast(IntegerType()))
    df_pred = df_pred.withColumn(TIMESTAMP_COL, lit(int(time.time())))

    # -------------------------
    # 7. Minimal output for downstream systems
    # -------------------------
    df_pred_final = df_pred.select(
        "transaction_id",
        PRED_COL,
        PROB_COL,
        TIMESTAMP_COL
    )

    # create a folder for today
    today_str = datetime.now().strftime("%Y-%m-%d")
    OUTPUT_PATH_DAILY = f"{output_path}date={today_str}/"

    logger.info(f"Writing predictions to: {OUTPUT_PATH_DAILY}")
    df_pred_final.write.mode("overwrite").parquet(OUTPUT_PATH_DAILY)

    # -------------------------
    # 8. Log run info to MLflow
    # -------------------------
    with mlflow.start_run(run_name=f"batch_inference_{int(time.time())}"):
        mlflow.log_param("model_uri", model_uri)
        mlflow.log_param("input_path", input_path)
        mlflow.log_param("output_path", output_path)

        total_rows = df_pred_final.count()
        fraud_count = df_pred_final.filter(col(PRED_COL) == 1).count()

        mlflow.log_metric("row_count", total_rows)
        mlflow.log_metric("predicted_fraud_ratio", fraud_count / total_rows if total_rows else 0)

    logger.info(f"✅ Inference complete: {total_rows} rows, {fraud_count} predicted frauds.")

    spark.stop()
