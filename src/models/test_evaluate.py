import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

import sys
import numpy as np
import mlflow
from sklearn.ensemble import GradientBoostingRegressor

sys.path.append("src/data")
sys.path.append("src/features")
sys.path.append("src/models")

from data_ingestion import load_training_data
from preprocessing import add_rul_column
from data_split import split_by_engine
from feature_selection import select_features
from evaluate import evaluate_model


DATA_PATH = "data/raw/CMaps/train_FD001.txt"


def test_evaluate_model(tmp_path):
    # Load data
    df = load_training_data(DATA_PATH)

    # Add RUL
    df = add_rul_column(df)

    # Split by engine
    train_df, validation_df = split_by_engine(df)

    # Select features
    X_train, feature_names = select_features(train_df)
    X_validation, _ = select_features(validation_df)

    y_train = train_df["RUL"]
    y_validation = validation_df["RUL"]

    # Train a small temporary model for testing
    model = GradientBoostingRegressor(
        n_estimators=20,
        learning_rate=0.05,
        max_depth=2,
        random_state=42,
    )

    model.fit(X_train, y_train)

    # Temporary output files created by pytest
    metrics_path = tmp_path / "metrics.json"
    predictions_path = tmp_path / "predictions.csv"

    # Disable MLflow network logging during the test
    mlflow.set_tracking_uri("file:///tmp/mlruns-test")

    with mlflow.start_run():
        metrics = evaluate_model(
            model,
            X_validation,
            y_validation,
            str(metrics_path),
            str(predictions_path),
        )

    # Check expected metrics
    assert "MAE" in metrics
    assert "RMSE" in metrics
    assert "R2" in metrics

    # Check metrics are valid numbers
    assert np.isfinite(metrics["MAE"])
    assert np.isfinite(metrics["RMSE"])
    assert np.isfinite(metrics["R2"])

    # Basic sanity checks
    assert metrics["MAE"] >= 0
    assert metrics["RMSE"] >= 0

    # Check output files were created
    assert metrics_path.exists()
    assert predictions_path.exists()