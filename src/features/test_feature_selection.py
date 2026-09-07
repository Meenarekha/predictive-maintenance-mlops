import sys

sys.path.append("src/data")
sys.path.append("src/features")

from data_ingestion import load_training_data
from preprocessing import add_rul_column
from feature_selection import select_features

DATA_PATH = "data/raw/CMaps/train_FD001.txt"


def test_select_features():
    # Load data
    df = load_training_data(DATA_PATH)

    # Add RUL
    df = add_rul_column(df)

    # Select model features
    X, feature_names = select_features(df)

    # Check number of rows
    assert X.shape[0] == 20631

    # Check expected number of features
    assert X.shape[1] == 18
    assert len(feature_names) == 18

    # Check that engine ID is not used
    assert "engine" not in feature_names

    # Check that target is not used
    assert "RUL" not in feature_names

    # Check that helper column is not used
    assert "max_cycle" not in feature_names

    # Check constant features were removed
    constant_features = [
        "setting_3",
        "fan_inlet_temperature",
        "fan_inlet_pressure",
        "engine_pressure_ratio",
        "burner_fuel_air_ratio",
        "required_fan_speed",
        "required_fan_conversion_speed",
    ]

    for feature in constant_features:
        assert feature not in feature_names

    # Check there are no missing values
    assert X.isnull().sum().sum() == 0