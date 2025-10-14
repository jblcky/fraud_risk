from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
import mlflow
import mlflow.spark
import logging
import os

os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://minio:9000'
os.environ['AWS_ACCESS_KEY_ID'] = 'minio'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'minio123'

# Reduce Spark logging
logging.getLogger("py4j").setLevel(logging.WARNING)
logging.getLogger("pyspark").setLevel(logging.WARNING)

spark = SparkSession.builder.appName("FraudDetectionModelTraining").getOrCreate()

# CONFIGURE HADOOP FOR S3A (for Spark reading)
hadoop_conf = spark._jsc.hadoopConfiguration()
hadoop_conf.set("fs.s3a.access.key", "minio")
hadoop_conf.set("fs.s3a.secret.key", "minio123")
hadoop_conf.set("fs.s3a.endpoint", "http://minio:9000")
hadoop_conf.set("fs.s3a.path.style.access", "true")
hadoop_conf.set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
hadoop_conf.set("fs.s3a.connection.ssl.enabled", "false")

# Set Spark log level to WARN (hides INFO/debug logs)
spark.sparkContext.setLogLevel("WARN")

mlflow.set_tracking_uri("http://mlflow:5000")  # or your MLflow server
mlflow.set_experiment("FraudDetection")

df = spark.read.parquet("s3a://mlflow-bucket/model_ready/")

df = df.select("scaled_features", "rapid_transaction")

train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

with mlflow.start_run(run_name="logistic_regression_v1.0"):
    lr = LogisticRegression(featuresCol="scaled_features", labelCol="rapid_transaction", maxIter=10)
    model = lr.fit(train_df)

    preds = model.transform(test_df)
    binary_eval = BinaryClassificationEvaluator(labelCol="rapid_transaction", metricName="areaUnderROC")
    auc = binary_eval.evaluate(preds)

    precision_eval = MulticlassClassificationEvaluator(labelCol="rapid_transaction", predictionCol="prediction", metricName="precisionByLabel")
    recall_eval = MulticlassClassificationEvaluator(labelCol="rapid_transaction", predictionCol="prediction", metricName="recallByLabel")
    f1_eval = MulticlassClassificationEvaluator(labelCol="rapid_transaction", predictionCol="prediction", metricName="f1")

    precision = precision_eval.evaluate(preds)
    recall = recall_eval.evaluate(preds)
    f1 = f1_eval.evaluate(preds)

    print(f"AUC: {auc:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")

    # 8️⃣ Log everything to MLflow
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_metric("AUC", auc)
    mlflow.log_metric("Precision", precision)
    mlflow.log_metric("Recall", recall)
    mlflow.log_metric("F1", f1)

    mlflow.spark.log_model(model, "spark-model")
