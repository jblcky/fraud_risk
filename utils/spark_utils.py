from pyspark.sql import SparkSession
import logging

logger = logging.getLogger(__name__)

def get_spark_session(app_name="SparkApp"):
    builder = SparkSession.builder.appName(app_name) \
        .config("spark.sql.shuffle.partitions", "4") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .config("spark.sql.session.timeZone", "Asia/Singapore") \
        .config("spark.jars", "/opt/spark/jars/hadoop-aws-3.3.4.jar,"
                          "/opt/spark/jars/hadoop-common-3.3.4.jar,"
                          "/opt/spark/jars/aws-java-sdk-bundle-1.12.262.jar") \
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
        .config("spark.hadoop.fs.s3a.endpoint", "http://minio:9000")

    try:
        spark = builder.getOrCreate()
    except Exception as e:
        logger.error(f"Failed to create SparkSession: {e}")
        raise

    hadoop_conf = spark._jsc.hadoopConfiguration()
    hadoop_conf.set("fs.s3a.access.key", "minio")
    hadoop_conf.set("fs.s3a.secret.key", "minio123")
    hadoop_conf.set("fs.s3a.path.style.access", "true")
    hadoop_conf.set("fs.s3a.connection.ssl.enabled", "false")
    hadoop_conf.set("fs.s3a.committer.name", "directory")
    hadoop_conf.set("mapreduce.fileoutputcommitter.marksuccessfuljobs", "false")

    spark.sparkContext.setLogLevel("WARN")
    return spark
