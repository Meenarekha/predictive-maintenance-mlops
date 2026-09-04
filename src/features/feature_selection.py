import pandas as pd


CONSTANT_FEATURES = [
    "setting_3",
    "fan_inlet_temperature",
    "fan_inlet_pressure",
    "engine_pressure_ratio",
    "burner_fuel_air_ratio",
    "required_fan_speed",
    "required_fan_conversion_speed",
]


EXCLUDED_COLUMNS = [
    "engine",
    "max_cycle",
    "RUL",
]


def select_features(
    df: pd.DataFrame
) -> tuple[pd.DataFrame, list[str]]:
    """
    Select the features used by the RUL prediction model.

    Removes:
    - constant features
    - engine identifier
    - helper/leakage columns
    - target column

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed training dataframe.

    Returns
    -------
    tuple
        Selected feature dataframe and feature names.
    """

    columns_to_remove = (
        CONSTANT_FEATURES + EXCLUDED_COLUMNS
    )

    available_columns_to_remove = [
        column
        for column in columns_to_remove
        if column in df.columns
    ]

    X = df.drop(
        columns=available_columns_to_remove
    )

    feature_names = X.columns.tolist()

    return X, feature_names