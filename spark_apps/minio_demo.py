from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("MinIO Demo") \
    .config("spark.hadoop.fs.s3a.access.key", "minio") \
    .config("spark.hadoop.fs.s3a.secret.key", "minio123") \
    .config("spark.hadoop.fs.s3a.endpoint", "http://minio:9000") \
    .config("spark.hadoop.fs.s3a.path.style.access", "true") \
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
    .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false") \
    .getOrCreate()

# Write a sample DataFrame to MinIO
df = spark.range(10).toDF("id")
df.write.mode("overwrite").parquet("s3a://mlflow-bucket/demo/output.parquet")

print("✅ Successfully wrote to MinIO!")
print("Check MinIO console at http://localhost:9001")
