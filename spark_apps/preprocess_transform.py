from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, hour, dayofweek, count, window, lit
from pyspark.ml.feature import StringIndexer, StandardScaler, VectorAssembler, Imputer
from pyspark.ml import Pipeline
import logging


# Reduce Spark logging
logging.getLogger("py4j").setLevel(logging.WARNING)
logging.getLogger("pyspark").setLevel(logging.WARNING)


spark = SparkSession.builder.appName("PreprocessTransform").getOrCreate()


spark._jsc.hadoopConfiguration().set("fs.s3a.access.key", "minio")
spark._jsc.hadoopConfiguration().set("fs.s3a.secret.key", "minio123")
spark._jsc.hadoopConfiguration().set("fs.s3a.endpoint", "http://minio:9000")
spark._jsc.hadoopConfiguration().set("fs.s3a.path.style.access", "true")
spark._jsc.hadoopConfiguration().set("fs.s3a.impl","org.apache.hadoop.fs.s3a.S3AFileSystem")
spark._jsc.hadoopConfiguration().set("fs.s3a.connection.ssl.enabled", "false")

# Set Spark log level to WARN (hides INFO/debug logs)
spark.sparkContext.setLogLevel("WARN")

# 1. LOAD RAW DATA
print("📥 Loading raw transactions...")
df = spark.read.parquet("s3a://mlflow-bucket/transactions/")

# Show schema and sample
df.printSchema()
df.show(5)
print(f"Total records: {df.count()}")

df = df.dropna(subset=["transaction_id", "user_id", "amount", "timestamp_ts"])

'''
In many real-world ML tasks, time-of-day affects behavior.
Breaking time into interpretable units (hour, day of week, weekday/weekend, etc.) creates meaningful features.
Example: users may spend more on weekends, or fraud may spike on certain days.
ML models can now use temporal patterns to improve predictions.
hour_of_day → captures daily patterns
day_of_week → captures weekly patterns
'''
df = df.withColumn(
    "hour_of_day",
    when(col("timestamp_ts").isNotNull(), hour(col("timestamp_ts"))).otherwise(0)
)

df = df.withColumn(
    "day_of_week",
    when(col("timestamp_ts").isNotNull(), dayofweek(col("timestamp_ts"))).otherwise(0)
)

'''
type_index → converts categorical info to numeric, making it usable for ML
'''
df = df.withColumn("type_index", when(col("type") == "purchase", 1).otherwise(0))

user_tx_count = df.groupBy(
    window(col("timestamp_ts"), "5 minutes"),  # 5-minute rolling window
    col("user_id")
).agg(count("*").alias("user_tx_count"))

'''
binary feature: 1 = user did more than 5 transactions in 5 minutes, 0 = otherwise.
'''
user_tx_count = user_tx_count.withColumn(
    "rapid_transaction",
    when(col("user_tx_count") > 5, 1).otherwise(0)
)

df = df.join(user_tx_count, on="user_id", how="left")

'''
Fill missing values with defaults (recommended for ML) after joins.
'''
df = df.fillna({"user_tx_count": 0, "rapid_transaction": 0})

'''
use an imputer to replace missing values with the median of the column.
This is better than mean for skewed data, as it reduces the impact of outliers.
'''
# Check if columns have non-null values before imputing
if df.filter(col("amount").isNotNull()).count() > 0:
    imputer = Imputer(
        inputCols=["amount", "user_tx_count"],
        outputCols=["amount_imputed", "user_tx_count_imputed"]
    ).setStrategy("median")
    df = imputer.fit(df).transform(df)
else:
    # Fallback: use 0 if all values are null
    df = df.withColumn("amount_imputed", lit(0.0)) \
           .withColumn("user_tx_count_imputed", lit(0))

'''
sanity check for nulls after imputation before scaling
'''
df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns]).show()
df.printSchema()
df.show(5)
print(f"Total records: {df.count()}")

# Assemble features into a vector
assembler = VectorAssembler(
    inputCols=["amount_imputed", "user_tx_count_imputed", "hour_of_day", "day_of_week", "type_index", "rapid_transaction"],
    outputCol="features_vector"
)

# Apply StandardScaler
scaler = StandardScaler(inputCol="features_vector", outputCol="scaled_features")
pipeline = Pipeline(stages=[assembler, scaler])

# Fit and transform the data
pipeline_model = pipeline.fit(df)
df_scaled = pipeline_model.transform(df)
df_scaled.printSchema()
df_scaled.show(5)

df_scaled.write.mode("overwrite").parquet("s3a://mlflow-bucket/model_ready/")
