import os
import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import GradientBoostingRegressor


def train_model(X_train, y_train, model_path):
    """
    Train the selected Gradient Boosting model,
    save it locally, and log parameters/model to MLflow.
    """

    # ---------------------------------------------------------
    # Create the selected model
    # ---------------------------------------------------------

    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )

    # ---------------------------------------------------------
    # Log model parameters to the active MLflow run
    # ---------------------------------------------------------

    mlflow.log_params({
        "model_type": "GradientBoostingRegressor",
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 3,
        "random_state": 42
    })

    # ---------------------------------------------------------
    # Train model
    # ---------------------------------------------------------

    model.fit(
        X_train,
        y_train
    )

    # ---------------------------------------------------------
    # Log trained model to MLflow
    # ---------------------------------------------------------

    mlflow.sklearn.log_model(
        model,
        name="gradient_boosting_model"
    )

    # ---------------------------------------------------------
    # Save model locally
    # ---------------------------------------------------------

    os.makedirs(
        os.path.dirname(model_path),
        exist_ok=True
    )

    joblib.dump(
        model,
        model_path
    )

    return model