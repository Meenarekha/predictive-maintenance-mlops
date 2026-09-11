import os
import sys

import pandas as pd

sys.path.append("src/data")
sys.path.append("src/features")

from data_ingestion import load_training_data
from feature_selection import select_features
from drift_detection import monitor_features


DATA_PATH = "data/raw/CMaps/train_FD001.txt"
OUTPUT_PATH = "results/drift_report.csv"


def main():
    # Load reference/training data
    df = load_training_data(DATA_PATH)

    # Select the same features used by the model
    reference_data, feature_names = select_features(df)

    # ---------------------------------------------------------
    # Demo current data
    # ---------------------------------------------------------
    # In production, this would be replaced by recent
    # prediction/input data from the API.
    #
    # For now, create a slightly modified copy so that
    # the monitoring pipeline can be demonstrated.
    current_data = reference_data.copy()

    # Introduce a small controlled change to demonstrate monitoring
    current_data["cycle"] = current_data["cycle"] * 1.05

    # Calculate feature drift
    report = monitor_features(
        reference_data,
        current_data,
        feature_names
    )

    # Create output directory
    os.makedirs(
        os.path.dirname(OUTPUT_PATH),
        exist_ok=True
    )

    # Save report
    report.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\n===== MODEL DRIFT REPORT =====\n")
    print(report.to_string(index=False))

    print(f"\nReport saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()