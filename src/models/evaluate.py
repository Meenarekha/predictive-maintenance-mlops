import os
import json
import pandas as pd
import mlflow
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_model(
    model,
    X_validation,
    y_validation,
    metrics_path,
    predictions_path
):
    """
    Evaluate the trained model, save metrics and predictions,
    and log the metrics to MLflow.
    """

    # Make predictions
    predictions = model.predict(X_validation)

    # Calculate metrics
    mae = mean_absolute_error(y_validation, predictions)
    rmse = mean_squared_error(y_validation, predictions) ** 0.5
    r2 = r2_score(y_validation, predictions)

    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }

    # Save metrics locally
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    # Save predictions locally
    os.makedirs(os.path.dirname(predictions_path), exist_ok=True)

    predictions_df = pd.DataFrame({
        "actual_RUL": y_validation,
        "predicted_RUL": predictions
    })

    predictions_df.to_csv(predictions_path, index=False)

    # Log metrics to the active MLflow run
    mlflow.log_metrics({
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    return metrics