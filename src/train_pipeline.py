import os
import sys

# Allow imports from src subdirectories
sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "data"
    )
)

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "features"
    )
)

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "models"
    )
)

from data_ingestion import load_training_data
from preprocessing import add_rul_column
from data_split import split_by_engine
from feature_selection import select_features
from train import train_model
from evaluate import evaluate_model


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

DATA_PATH = "data/raw/CMaps/train_FD001.txt"

MODEL_PATH = "models/final_gradient_boosting.pkl"

METRICS_PATH = (
    "results/final_model_metrics.json"
)

PREDICTIONS_PATH = (
    "results/final_model_predictions.csv"
)


# ---------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------

def run_pipeline():

    print("=" * 60)
    print("PREDICTIVE MAINTENANCE TRAINING PIPELINE")
    print("=" * 60)

    # -----------------------------------------------------
    # Step 1: Data ingestion
    # -----------------------------------------------------

    print("\n[1/6] Loading data...")

    df = load_training_data(
        DATA_PATH
    )

    print(f"Loaded data: {df.shape}")


    # -----------------------------------------------------
    # Step 2: Preprocessing
    # -----------------------------------------------------

    print("\n[2/6] Adding RUL...")

    df = add_rul_column(
        df
    )

    print("RUL calculated successfully.")


    # -----------------------------------------------------
    # Step 3: Train/validation split
    # -----------------------------------------------------

    print("\n[3/6] Splitting by engine...")

    train_df, validation_df = split_by_engine(
        df
    )

    print(
        f"Training engines: "
        f"{train_df['engine'].nunique()}"
    )

    print(
        f"Validation engines: "
        f"{validation_df['engine'].nunique()}"
    )


    # -----------------------------------------------------
    # Step 4: Feature selection
    # -----------------------------------------------------

    print("\n[4/6] Selecting features...")

    X_train, feature_names = select_features(
        train_df
    )

    X_validation, _ = select_features(
        validation_df
    )

    y_train = train_df["RUL"]

    y_validation = validation_df["RUL"]

    print(
        f"Number of features: "
        f"{len(feature_names)}"
    )


    # -----------------------------------------------------
    # Step 5: Train model
    # -----------------------------------------------------

    print("\n[5/6] Training Gradient Boosting model...")

    model = train_model(
        X_train,
        y_train,
        MODEL_PATH
    )

    print(
        f"Model saved to: "
        f"{MODEL_PATH}"
    )


    # -----------------------------------------------------
    # Step 6: Evaluate
    # -----------------------------------------------------

    print("\n[6/6] Evaluating model...")

    metrics = evaluate_model(
        model,
        X_validation,
        y_validation,
        METRICS_PATH,
        PREDICTIONS_PATH
    )

    print("\nFinal metrics:")

    for metric, value in metrics.items():
        print(
            f"{metric}: {value:.4f}"
        )

    print(
        f"\nMetrics saved to: "
        f"{METRICS_PATH}"
    )

    print(
        f"Predictions saved to: "
        f"{PREDICTIONS_PATH}"
    )

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    run_pipeline()