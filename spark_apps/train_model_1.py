def train_model(mlflow_uri: str, input_path: str, artifact_path="xgb_model", registered_model_name="fraud_xgb_model"):
    from pyspark.sql import SparkSession
    import pandas as pd
    import xgboost as xgb
    from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, precision_recall_curve
    import mlflow
    import mlflow.xgboost
    from mlflow.exceptions import RestException
    import io
    import joblib
    import boto3
    from collections import Counter
    from datetime import datetime
    import sys
    import numpy as np
    import logging
    from utils.spark_utils import get_spark_session

    # MLFLOW_URI = "http://mlflow:5000"

    logging.getLogger("py4j").setLevel(logging.WARNING)
    logging.getLogger("pyspark").setLevel(logging.WARNING)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # -------------------------------
    # 1️⃣ Initialize Spark
    # -------------------------------
    spark = get_spark_session(app_name="XGBoostFraudMLflow")

    # -------------------------------
    # 2️⃣ Load processed features from MinIO
    # -------------------------------
    # data_path = "s3a://mlflow-bucket/model_ready/"
    df = spark.read.parquet(input_path)
    df.cache()
    pdf = df.toPandas()

    feature_cols = [
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

    target_col = "is_fraudulent"

    X = pdf[feature_cols]
    y = pdf[target_col]

    # -------------------------------
    # 3️⃣ Train/test split
    # -------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # -------------------------------
    # 5️⃣ Handle class imbalance
    # -------------------------------
    counter = Counter(y_train)
    scale_pos_weight = counter[0] / counter[1]  # imbalance ratio

    # -------------------------------
    # 4️⃣ Set MLflow tracking
    # -------------------------------
    mlflow.set_tracking_uri(mlflow_uri)  # MLflow server container URL
    mlflow.set_experiment("fraud_detection_xgboost")

    with mlflow.start_run(run_name="xgb_fraud_run") as run:
        # Log environment info for reproducibility
        mlflow.log_param("python_version", sys.version)
        mlflow.log_param("pyspark_version", spark.version)
        mlflow.log_param("pandas_version", pd.__version__)
        mlflow.log_param("xgboost_version", xgb.__version__)
        mlflow.log_param("features", feature_cols)
        mlflow.log_param("scale_pos_weight", scale_pos_weight)

        # -------------------------------
        # 7️⃣ Define XGBoost and hyperparameter search
        # -------------------------------
        xgb_clf = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='auc',
            random_state=42,
            scale_pos_weight=scale_pos_weight
        )

        param_dist = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 4, 5],
            'learning_rate': [0.05, 0.1, 0.2],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'gamma': [0, 0.1, 0.5, 1],
            'min_child_weight': [1, 3, 5]
        }

        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        rnd_search = RandomizedSearchCV(
            xgb_clf,
            param_distributions=param_dist,
            n_iter=20,
            scoring='f1',  # Optimize for F1 score
            n_jobs=-1,
            cv=skf,
            verbose=1,
            random_state=42
        )

        rnd_search.fit(X_train, y_train)
        best_model = rnd_search.best_estimator_
        mlflow.log_params(rnd_search.best_params_)

        # -------------------------------
        # 8️⃣ Automatic threshold selection to maximize F1
        # -------------------------------
        y_prob = best_model.predict_proba(X_test)[:, 1]
        precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-6)

        best_thresh = thresholds[np.argmax(f1_scores)]
        y_pred = (y_prob >= best_thresh).astype(int)
        mlflow.log_param("best_threshold", float(best_thresh))

        # -------------------------------
        # 6️⃣ Evaluate
        # -------------------------------
        metrics = {
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_prob),
            "accuracy": (y_test==y_pred).mean()
        }

        cm = confusion_matrix(y_test, y_pred)
        fi = best_model.feature_importances_

        # Convert to a dictionary with explicit actual/predicted keys
        cm_named = {
            "actual_non_fraud": {"predicted_non_fraud": int(cm[0,0]), "predicted_fraud": int(cm[0,1])},
            "actual_fraud": {"predicted_non_fraud": int(cm[1,0]), "predicted_fraud": int(cm[1,1])}
    }
        mlflow.log_dict({"confusion_matrix": cm_named}, "confusion_matrix.json")


        feature_map = {i: name for i, name in enumerate(feature_cols)}
        mlflow.log_dict(feature_map, "feature_map.json")

        # 2️⃣ Feature importance with names
        fi_named = {name: float(imp) for name, imp in zip(feature_cols, fi)}
        mlflow.log_dict(fi_named, "feature_importance_named.json")


        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        logger.info("✅ Metrics logged to MLflow:", metrics)

        # -------------------------------
        # 7️⃣ Log XGBoost model to MLflow
        # -------------------------------
        mlflow.xgboost.log_model(best_model, artifact_path='xgb_model', registered_model_name='fraud_xgb_model')
        logger.info("✅ Model logged to MLflow and registered")

    # -------------------------------
    # 8️⃣ Save model to MinIO directly
    # -------------------------------
    s3 = boto3.client(
        "s3",
        endpoint_url="http://minio:9000",
        aws_access_key_id="minio",
        aws_secret_access_key="minio123",
        region_name="us-east-1"
    )

    bucket_name = "mlflow-bucket"
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_key = f"models/xgb_fraud_model_{timestamp}.pkl"

    buffer = io.BytesIO()
    joblib.dump(best_model, buffer)
    buffer.seek(0)

    try:
        s3.upload_fileobj(buffer, bucket_name, model_key)
        logger.info(f"✅ Model uploaded directly to MinIO: s3a://{bucket_name}/{model_key}")
    except Exception as e:
        logger.error(f"❌ Failed to upload model to MinIO: {e}")
