import sys
import os
import json
import joblib

sys.path.append("src/data")
sys.path.append("src/features")
sys.path.append("src/models")

from data_ingestion import load_training_data
from preprocessing import add_rul_column
from data_split import split_by_engine
from feature_selection import select_features
from evaluate import evaluate_model


DATA_PATH = "data/raw/CMaps/train_FD001.txt"

MODEL_PATH = "models/test_gradient_boosting.pkl"

METRICS_PATH = "results/test_evaluation_metrics.json"

PREDICTIONS_PATH = "results/test_evaluation_predictions.csv"


def test_evaluate_model():
    # Load data
    df = load_training_data(DATA_PATH)

    # Add RUL
    df = add_rul_column(df)

    # Split by engine
    train_df, validation_df = split_by_engine(df)

    # Select features
    X_train, feature_names = select_features(train_df)
    X_validation, _ = select_features(validation_df)

    y_validation = validation_df["RUL"]

    # Load trained model
    model = joblib.load(MODEL_PATH)

    # Evaluate model
    metrics = evaluate_model(
        model,
        X_validation,
        y_validation,
        METRICS_PATH,
        PREDICTIONS_PATH
    )

    # Verify metrics exist
    assert "MAE" in metrics
    assert "RMSE" in metrics
    assert "R2" in metrics

    # Verify metric values are valid
    assert metrics["MAE"] >= 0
    assert metrics["RMSE"] >= 0
    assert 0 <= metrics["R2"] <= 1

    # Verify expected model performance
    assert metrics["MAE"] < 30
    assert metrics["RMSE"] < 35
    assert metrics["R2"] > 0.70

    # Verify metrics file exists
    assert os.path.exists(METRICS_PATH)

    # Verify predictions file exists
    assert os.path.exists(PREDICTIONS_PATH)

    # Verify metrics file contains valid JSON
    with open(METRICS_PATH, "r") as f:
        saved_metrics = json.load(f)

    assert saved_metrics["MAE"] == metrics["MAE"]
    assert saved_metrics["RMSE"] == metrics["RMSE"]
    assert saved_metrics["R2"] == metrics["R2"]