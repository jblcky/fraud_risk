from pyspark.sql import SparkSession
# Functions for JSON parsing, column manipulation, and windowing
from pyspark.sql.functions import (
    from_json, col, when, window, count, to_timestamp,
    lit, avg, sum as spark_sum, coalesce # Added coalesce for robust joining
)
# Data Types
from pyspark.sql.types import StructType, StringType, DoubleType, IntegerType, BooleanType
import threading
import logging

# Reduce Spark logging
logging.getLogger("py4j").setLevel(logging.WARNING)
logging.getLogger("pyspark").setLevel(logging.WARNING)

# --- 1. Define Schema (user_id changed to StringType for reliable parsing) ---
schema = StructType() \
    .add("transaction_id", StringType()) \
    .add("user_id", StringType()) \
    .add("amount", DoubleType()) \
    .add("currency", StringType()) \
    .add("country", StringType()) \
    .add("type", StringType()) \
    .add("timestamp", StringType()) \

# --- 2. Create Spark Session ---
spark = SparkSession.builder \
    .appName("KafkaTransactionConsumer") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# MinIO/S3A Configuration (Keep this for the write streams)
spark._jsc.hadoopConfiguration().set("fs.s3a.access.key", "minio")
spark._jsc.hadoopConfiguration().set("fs.s3a.secret.key", "minio123")
spark._jsc.hadoopConfiguration().set("fs.s3a.endpoint", "http://minio:9000")
spark._jsc.hadoopConfiguration().set("fs.s3a.path.style.access", "true")
spark._jsc.hadoopConfiguration().set("fs.s3a.impl","org.apache.hadoop.fs.s3a.S3AFileSystem")
spark._jsc.hadoopConfiguration().set("fs.s3a.connection.ssl.enabled", "false")

# --- 3. Read Stream from Kafka ---
df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "kafka:29092") \
    .option("subscribe", "transactions_fraud") \
    .option("startingOffsets", "earliest") \
    .option("failOnDataLoss", "false") \
    .load()

# Convert Kafka value and parse JSON
df = df.selectExpr("CAST(value AS STRING)")
df = df.select(from_json(col("value"), schema).alias("data")).select("data.*")

# --- 4. Data Cleaning & Feature Engineering (Static Features) ---

# Convert timestamp string to Spark TimestampType
df = df.withColumn("event_time", to_timestamp(col("timestamp"), "yyyy-MM-dd HH:mm:ss.SSSSSS"))

# Feature 1: High Value Check (Simple Threshold)
df = df.withColumn(
    "is_high_value",
    when(col("amount") > 400, lit(1)).otherwise(lit(0))
).withColumn( # Add a Boolean version for the Join below
    "is_high_value_flag",
    when(col("amount") > 400, True).otherwise(False)
)

# Feature 2: Foreign Transaction Check
df = df.withColumn(
    "is_foreign_tx",
    when(
        (col("currency") != "USD") & (~col("country").isin("USA", "UK")),
        lit(1)
    ).otherwise(lit(0))
).withColumn( # Add a Boolean version for the Join below
    "is_foreign_tx_flag",
    when(
        (col("currency") != "USD") & (~col("country").isin("USA", "UK")),
        True
    ).otherwise(False)
)

# --- 5. Rapid Transaction Detection (Velocity Check - Aggregate Stream) ---

# Alias the base stream (df) for use in the join later
tx_df = df.alias("tx")

# The aggregate stream must be defined before the join
user_tx_count = tx_df \
    .withWatermark("event_time", "10 minutes") \
    .groupBy(
        # The window must be part of the join key
        window(col("event_time"), "5 seconds", "1 seconds").alias("tx_window"), # Sliding window helps with join
        col("user_id")
    ) \
    .count() \
    .withColumnRenamed("count", "tx_count")

# Feature 3: Rapid Transaction Check (on the aggregated stream)
user_tx_count = user_tx_count.withColumn(
    "is_rapid_tx",
    when(col("tx_count") > 2, lit(1)).otherwise(lit(0))
).withColumn(
    "is_rapid_tx_flag",
    when(col("tx_count") > 2, True).otherwise(False)
)

# --- 6. Stream-Stream Join and Fraud Scoring ---

# Alias the aggregate stream
agg_df = user_tx_count.alias("agg")

# 🛑 CRITICAL STEP: Join the raw transaction stream (df) with the aggregate stream (user_tx_count)
# The join condition MUST include the key (user_id) AND the window bounds.
# A left outer join ensures transactions without a velocity match are still included.
final_fraud_df = tx_df.join(
    agg_df,
    (tx_df["user_id"] == agg_df["user_id"]) &
    (tx_df["event_time"] >= agg_df["tx_window"].start) &
    (tx_df["event_time"] <= agg_df["tx_window"].end),
    "leftOuter"
)

# Clean up joined columns and fill nulls from leftOuter join
final_fraud_df = final_fraud_df.select(
    # Use qualified names (e.g., from the 'tx' alias)
    col("tx.transaction_id").alias("transaction_id"),
    col("tx.user_id").alias("user_id"),
    col("tx.amount").alias("amount"),
    col("tx.currency").alias("currency"),
    col("tx.country").alias("country"),
    col("tx.type").alias("type"),
    col("tx.event_time").alias("event_time"),

    # Static features also come from the 'tx' alias
    col("tx.is_high_value"),
    col("tx.is_foreign_tx"),

    # Velocity features come from the 'agg' alias
    coalesce(col("agg.tx_count"), lit(1)).alias("tx_count_5s"),
    coalesce(col("agg.is_rapid_tx"), lit(0)).alias("is_rapid_tx"),
    coalesce(col("agg.is_rapid_tx_flag"), lit(False)).alias("is_rapid_tx_flag")
)

# Final Feature: Calculate Fraud Risk Score (Sum of Boolean Flags)
final_fraud_df = final_fraud_df.withColumn(
    "fraud_risk_score",
    col("is_high_value") + col("is_foreign_tx") + col("is_rapid_tx")
)

# Final Feature: Simple Fraud Determination (Score > 1)
final_fraud_df = final_fraud_df.withColumn(
    "is_fraudulent",
    when(col("fraud_risk_score") >= 2, True).otherwise(False) # 2 or more flags means likely fraud
)


# --- 7. Output Streams ---

# Stream 1: Save ALL FEATURED DATA to MinIO/S3A (Using the final DataFrame)
minio_query = final_fraud_df.writeStream \
    .format("parquet") \
    .option("checkpointLocation", "/opt/spark_apps/checkpoint_final_features") \
    .option("path", "s3a://mlflow-bucket/transactions_features/") \
    .outputMode("append") \
    .start()

# Stream 2: Console Output (for monitoring the final fraud score results)
# Only show the relevant columns for fraud detection
console_query = final_fraud_df.select(
    col("event_time"), col("user_id"), col("amount"), col("country"),
    col("is_high_value"), col("is_foreign_tx"), col("is_rapid_tx"),
    col("fraud_risk_score"), col("is_fraudulent")
) \
    .writeStream \
    .outputMode("append") \
    .format("console") \
    .option("truncate", False) \
    .start()

# --- 8. Await Termination ---
# Use threads to keep all streaming queries running simultaneously
threading.Thread(target=minio_query.awaitTermination).start()
console_query.awaitTermination()
