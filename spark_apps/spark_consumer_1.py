def kafka_consumer():
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import from_json, col, when, count, window, lit, avg, stddev, from_utc_timestamp
    from pyspark.sql.types import StructType, StringType, DoubleType, TimestampType

    MINIO_ENDPOINT = "http://minio:9000"
    CHECKPOINT_PATH = "s3a://mlflow-bucket/checkpoints/transactions_fraud"
    OUTPUT_PATH = "s3a://mlflow-bucket/transactions_features/"

    # Broadcast currency rates (so not re-shipped each microbatch)
    exchange_rates = {
        "USD": 1.0, "EUR": 1.1, "GBP": 1.25, "JPY": 0.0067, "SGD": 0.73, "MYR": 0.21
    }

    # --- 1. Schema ---
    schema = StructType() \
        .add("transaction_id", StringType()) \
        .add("user_id", StringType()) \
        .add("amount", DoubleType()) \
        .add("currency", StringType()) \
        .add("country", StringType()) \
        .add("type", StringType()) \
        .add("merchant", StringType()) \
        .add("timestamp_utc", TimestampType())

    # --- 2. Spark session ---
    spark = SparkSession.builder.appName("KafkaConsumer") \
            .config("spark.sql.shuffle.partitions", "4") \
            .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
            .config("spark.sql.session.timeZone", "Asia/Singapore") \
            .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    # --- 3. MinIO config ---
    spark._jsc.hadoopConfiguration().set("fs.s3a.access.key", "minio")
    spark._jsc.hadoopConfiguration().set("fs.s3a.secret.key", "minio123")
    spark._jsc.hadoopConfiguration().set("fs.s3a.endpoint", MINIO_ENDPOINT)
    spark._jsc.hadoopConfiguration().set("fs.s3a.path.style.access", "true")
    spark._jsc.hadoopConfiguration().set("fs.s3a.impl","org.apache.hadoop.fs.s3a.S3AFileSystem")
    spark._jsc.hadoopConfiguration().set("fs.s3a.connection.ssl.enabled", "false")
    spark._jsc.hadoopConfiguration().set("fs.s3a.committer.name", "directory")
    spark._jsc.hadoopConfiguration().set("mapreduce.fileoutputcommitter.marksuccessfuljobs", "false")

    # --- 4. Read Kafka ---
    df = spark.readStream.format("kafka") \
        .option("kafka.bootstrap.servers", "kafka:29092") \
        .option("subscribe", "transactions_fraud") \
        .option("startingOffsets", "earliest") \
        .option('failOnDataLoss', 'false') \
        .load()

    df = df.selectExpr("CAST(value AS STRING)").select(from_json(col("value"), schema).alias("data")).select("data.*")
    df = df.withColumn("event_time",
        from_utc_timestamp(col("timestamp_utc"), "Asia/Singapore")
    ).withWatermark("event_time", "10 seconds")

    # --- 5. Add static features ---
    df = df.withColumn("is_foreign_tx", when((col("currency") != "USD") & (~col("country").isin("USA","UK")), 1).otherwise(0))

    # Create amount_usd column using conditional logic
    # Currency conversion (broadcasted)
    df = df.withColumn(
        "amount_usd",
        when(col("currency") == "USD", col("amount") * lit(exchange_rates["USD"]))
        .when(col("currency") == "EUR", col("amount") * lit(exchange_rates["EUR"]))
        .when(col("currency") == "GBP", col("amount") * lit(exchange_rates["GBP"]))
        .when(col("currency") == "JPY", col("amount") * lit(exchange_rates["JPY"]))
        .when(col("currency") == "SGD", col("amount") * lit(exchange_rates["SGD"]))
        .when(col("currency") == "MYR", col("amount") * lit(exchange_rates["MYR"]))
        .otherwise(col("amount"))
    )

    df = df.withColumn("is_high_value", when(col("amount_usd") > 7000, 1).otherwise(0))
    df = df.withColumn("event_date", col("event_time").cast("date"))

    # --- 6. Use foreachBatch for windowed aggregation (no stream-stream join) ---
    def process_batch(batch_df, batch_id):
        # --- 6a. Rapid transaction counts ---
        try:
            batch_df.cache()

            rapid_df = batch_df.groupBy(
                "user_id",
                window("event_time", "5 seconds")
            ).agg(
                count("*").alias("tx_count_5s")
            ).withColumn("is_rapid_tx", when(col("tx_count_5s") > 2, 1).otherwise(0))

            batch_df = batch_df.join(
                rapid_df,
                (batch_df.user_id == rapid_df.user_id) &
                (batch_df.event_time >= rapid_df.window.start) &
                (batch_df.event_time <= rapid_df.window.end),
                "left"
            ).select(
                batch_df["*"],
                rapid_df.tx_count_5s,
                rapid_df.is_rapid_tx
            )

            # --- 6b. User behavior: avg and std of amount per user ---
            user_stats = batch_df.groupBy("user_id").agg(
                avg("amount").alias("avg_amount_user"),
                stddev("amount").alias("std_amount_user")
            )
            batch_df = batch_df.join(user_stats, on="user_id", how="left")

            # --- 6c. Merchant frequency per user ---
            merchant_stats = batch_df.groupBy("user_id", "merchant").agg(
                count("*").alias("merchant_count_user")
            )
            batch_df = batch_df.join(merchant_stats, on=["user_id","merchant"], how="left")
            batch_df = batch_df.withColumn("is_new_merchant", when(col("merchant_count_user") <= 1, 1).otherwise(0))

            # --- 6d. Amount anomaly ---
            k = 3  # threshold multiplier
            batch_df = batch_df.withColumn(
                "is_amount_anomaly",
                when(col("amount") > col("avg_amount_user") + k * col("std_amount_user"), 1).otherwise(0)
            )

            # --- 6e. Fraud risk score and label ---
            batch_df = batch_df.withColumn(
                "fraud_risk_score",
                col("is_high_value") + col("is_foreign_tx") + col("is_rapid_tx") + col("is_new_merchant") + col("is_amount_anomaly")
            ).withColumn(
                "is_fraudulent",
                (col("fraud_risk_score") >= 4).cast("int")
            )

            # --- 6f. Write features to MinIO ---
            batch_df.coalesce(1).write.mode("append").partitionBy("event_date").parquet(OUTPUT_PATH)

            # --- 6g. Show for debugging ---
            batch_df.select(
                "event_time","user_id","amount","is_high_value","is_foreign_tx",
                "is_rapid_tx","avg_amount_user","std_amount_user",
                "merchant","merchant_count_user","is_new_merchant",
                "is_amount_anomaly",
                "fraud_risk_score","is_fraudulent"
            ).show(truncate=False)

        except Exception as e:
            print(f"[ERROR] Batch {batch_id}: {e}")

        finally:
            batch_df.unpersist()

    # --- 7. Start streaming ---
    query = df.writeStream \
        .foreachBatch(process_batch) \
        .option("checkpointLocation", CHECKPOINT_PATH) \
        .start()

    try:
        query.awaitTermination()
    except KeyboardInterrupt:
        print("Stopping stream...")
        query.stop()
        spark.stop()
