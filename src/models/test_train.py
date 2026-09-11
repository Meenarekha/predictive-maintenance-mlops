import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

import sys
import mlflow

mlflow.set_tracking_uri("file:///tmp/mlruns-test")

sys.path.append("src/data")
sys.path.append("src/features")
sys.path.append("src/models")


from data_ingestion import load_training_data
from preprocessing import add_rul_column
from data_split import split_by_engine
from feature_selection import select_features
from train import train_model


DATA_PATH = "data/raw/CMaps/train_FD001.txt"
MODEL_PATH = "models/test_gradient_boosting.pkl"


def test_train_model():
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

    # Train model
    model = train_model(
        X_train,
        y_train,
        MODEL_PATH
    )

    # Verify training data
    assert X_train.shape == (16561, 18)
    assert X_validation.shape == (4070, 18)

    # Verify features
    assert len(feature_names) == 18

    # Verify model was created
    assert model is not None

    # Verify model can make predictions
    predictions = model.predict(X_validation)

    assert len(predictions) == len(y_validation)

    # Verify model file exists
    assert os.path.exists(MODEL_PATH)