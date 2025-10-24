from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum as spark_sum

# Spark session
spark = SparkSession.builder.appName("CheckParquetMinIO").getOrCreate()

# MinIO config
spark._jsc.hadoopConfiguration().set("fs.s3a.access.key", "minio")
spark._jsc.hadoopConfiguration().set("fs.s3a.secret.key", "minio123")
spark._jsc.hadoopConfiguration().set("fs.s3a.endpoint", "http://minio:9000")
spark._jsc.hadoopConfiguration().set("fs.s3a.path.style.access", "true")
spark._jsc.hadoopConfiguration().set("fs.s3a.impl","org.apache.hadoop.fs.s3a.S3AFileSystem")
spark._jsc.hadoopConfiguration().set("fs.s3a.connection.ssl.enabled", "false")

# Load the parquet folder
df = spark.read.parquet("s3a://mlflow-bucket/transactions_features/")

# Show schema
df.printSchema()

# Show first few rows
df.show(5, truncate=False)

# Count total rows
print(f"Total rows: {df.count()}")

# df is your DataFrame loaded from MinIO
null_summary = df.select([
    spark_sum(col(c).isNull().cast("int")).alias(c) for c in df.columns
])

# Show columns that have at least one null
cols_with_nulls = [c for c in null_summary.columns if null_summary.collect()[0][c] > 0]
print("Columns with nulls:", cols_with_nulls)
