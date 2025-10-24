def preprocess_transform(input_path, output_path):
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import col, hour, dayofweek
    import logging
    from utils.spark_utils import get_spark_session

    # input_path = "s3a://mlflow-bucket/transactions_features/"
    # output_path = "s3a://mlflow-bucket/model_ready/"

    # -----------------------------
    # Spark session
    # -----------------------------
    logging.getLogger("py4j").setLevel(logging.WARNING)
    logging.getLogger("pyspark").setLevel(logging.WARNING)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    spark = get_spark_session(app_name="MLFeatureEngineering")

    # -----------------------------
    # Load output from spark_consumer.py
    # -----------------------------
    df = spark.read.parquet(input_path)

    # -----------------------------
    # Temporal features
    # -----------------------------
    df = df.withColumn("hour_of_day", hour(col("event_time")))
    df = df.withColumn("day_of_week", dayofweek(col("event_time")))

    # -----------------------------
    # Selected features for XGBoost
    # -----------------------------
    selected_features = [
        "is_high_value",
        "is_foreign_tx",
        "is_rapid_tx",
        "is_new_merchant",
        "is_amount_anomaly",
        "hour_of_day",
        "day_of_week",
        "avg_amount_user",
        "std_amount_user",
        "is_fraudulent"  # label
    ]

    df = df.select(selected_features)

    # -----------------------------
    # Handle nulls
    # -----------------------------
    # Fill numeric nulls with 0
    numeric_cols = [c for c in df.columns if c != "is_fraudulent"]
    df = df.fillna({c: 0 for c in numeric_cols})

    # -----------------------------
    # Save cleaned data to MinIO
    # -----------------------------
    df.write.mode("overwrite").parquet(output_path)

    logger.info(f"Preprocessing completed. Output saved to {output_path}")
