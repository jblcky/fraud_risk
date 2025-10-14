from pyspark.sql import SparkSession
from pyspark.sql.functions import monotonically_increasing_id
import mlflow
import mlflow.spark
import logging
import os
import tempfile
from pyspark.ml.pipeline import PipelineModel
from mlflow.client import MlflowClient

# 1️⃣ SET MLFLOW CREDENTIALS (for boto3) (for artifact storage - MinIO)
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://minio:9000'
os.environ['AWS_ACCESS_KEY_ID'] = 'minio'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'minio123'


# Reduce Spark logging
logging.getLogger("py4j").setLevel(logging.WARNING)
logging.getLogger("pyspark").setLevel(logging.WARNING)

# 🔑 CONNECT TO MLFLOW FIRST (BEFORE ANYTHING ELSE)
mlflow.set_tracking_uri("http://mlflow:5000")

MODEL_URI = "runs:/94bf064236e1475b986f18f2db29ce85/spark-model"

# MANDATORY: Configure Spark's Hadoop environment for S3A (MinIO) access
# CRITICAL FIXES: Adding s3a.client.factory and disabling cache for executor propagation
s3a_config = {
    # 1. Credentials
    "fs.s3a.access.key": "minio",
    "fs.s3a.secret.key": "minio123",

    # 2. MinIO Endpoint (Hostname and Port only, standard S3A/MinIO practice)
    "fs.s3a.endpoint": "minio:9000",

    # 3. Path Style Access (Crucial for MinIO)
    "fs.s3a.path.style.access": "true",

    # 4. Implementation
    "fs.s3a.impl": "org.apache.hadoop.fs.s3a.S3AFileSystem",

    # 5. Disable SSL
    "fs.s3a.connection.ssl.enabled": "false",
}


# -----------------------------
# 1️⃣ STOP ANY EXISTING SPARK SESSION
# -----------------------------
# This ensures we can create a clean session
try:
    from pyspark.sql import SparkSession
    SparkSession.getActiveSession().stop()
except:
    pass  # No active session

# -----------------------------
# 1️⃣ Spark Session + MinIO config
# -----------------------------
builder = SparkSession.builder.appName("FraudBatchInference").config("spark.sql.adaptive.enabled", "false")

# Inject all S3A configurations directly into the builder
for key, value in s3a_config.items():
    builder = builder.config(key, value)

spark = builder.getOrCreate()

# Apply configuration via both methods for maximum compatibility
for key, value in s3a_config.items():
    spark.conf.set(key, value)
    spark._jsc.hadoopConfiguration().set(key, value)

# Set Spark log level to WARN (hides INFO/debug logs)
spark.sparkContext.setLogLevel("WARN")

# 2️⃣ LOAD MODEL DIRECTLY FROM S3A
try:
    print(f"[INFO] Resolving S3A path for model URI: {MODEL_URI}")

    # 1. Extract run_id and artifact_path
    uri_parts = MODEL_URI.split("/")
    run_id = uri_parts[1] # e.g., 94bf064236e1475b986f18f2db29ce85
    artifact_path = "/".join(uri_parts[2:]) # e.g., spark-model

    # 2. Use MlflowClient to reliably get the *run-specific* base artifact URI
    client = mlflow.MlflowClient()
    run = client.get_run(run_id)
    artifact_root_uri = run.info.artifact_uri # e.g., s3://mlflow-bucket/<run_id>/artifacts

    # 🔑 CRITICAL FIX: Ensure the URI uses the 's3a' scheme, not 's3',
    # as 's3' is often unsupported in containerized Spark environments.
    if artifact_root_uri.startswith("s3://"):
        artifact_root_uri = artifact_root_uri.replace("s3://", "s3a://", 1)

    # 3. Construct the final S3A path pointing to the 'sparkml' subdirectory of the model artifact
    # The full path should be s3a://<bucket>/<run_id>/artifacts/<artifact_path>/sparkml
    S3A_MODEL_PATH = f"{artifact_root_uri}/{artifact_path}/sparkml"

    print(f"[INFO] Full S3A Model Path resolved to: {S3A_MODEL_PATH}")

    # 4. Load the model directly using the explicit S3A path via Spark's native loader
    model = PipelineModel.load(S3A_MODEL_PATH)
    print("[INFO] Model (PipelineModel) loaded successfully from S3A.")
except Exception as e:
    # This block is essential for debugging
    print(f"[ERROR] Failed to load model from S3A path: {e}")
    raise e


# -----------------------------
# 2️⃣ Load new data (transformed features)
# -----------------------------
input_path = "s3a://mlflow-bucket/model_ready/"
df = spark.read.parquet(input_path)


# 🔑 SAFETY CHECK
if df.rdd.isEmpty():
    raise ValueError("No data found in model_ready!")
print(f"[INFO] Loaded {df.count()} rows for inference")


# -----------------------------
# 4️⃣ Make predictions
# -----------------------------

# Select only the feature column
inference_df = df.select("scaled_features")

# Model expects same input schema as training
predictions = model.transform(inference_df)

# Join back to original df for context (optional)
predictions = predictions.withColumn("row_index", monotonically_increasing_id())
df = df.withColumn("row_index", monotonically_increasing_id())

predictions = predictions.join(df, on="row_index", how="left").drop("row_index")

# Keep only useful columns
predictions = predictions.select(
    "transaction_id",
    "user_id",
    "amount",
    "rapid_transaction",
    "prediction"
)

predictions.printSchema()
predictions.show(5, truncate=False)
# -----------------------------
# 5️⃣ Save predictions back to MinIO
# -----------------------------
output_path = "s3a://mlflow-bucket/predictions/"
predictions.write.mode("overwrite").parquet(output_path)
print(f"Saved predictions to {output_path}")
