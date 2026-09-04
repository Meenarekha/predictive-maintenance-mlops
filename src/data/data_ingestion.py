import os
import pandas as pd


COLUMNS = [
    "engine",
    "cycle",
    "setting_1",
    "setting_2",
    "setting_3",
    "fan_inlet_temperature",
    "lpc_outlet_temperature",
    "hpc_outlet_temperature",
    "lpt_outlet_temperature",
    "fan_inlet_pressure",
    "bypass_duct_pressure",
    "hpc_outlet_pressure",
    "physical_fan_speed",
    "physical_core_speed",
    "engine_pressure_ratio",
    "hpc_outlet_static_pressure",
    "fuel_flow_to_ps30_ratio",
    "corrected_fan_speed",
    "corrected_core_speed",
    "bypass_ratio",
    "burner_fuel_air_ratio",
    "bleed_enthalpy",
    "required_fan_speed",
    "required_fan_conversion_speed",
    "high_pressure_turbine_cool_air_flow",
    "low_pressure_turbine_cool_air_flow",
]


def load_training_data(input_path: str) -> pd.DataFrame:
    """
    Load the raw C-MAPSS training dataset.

    Parameters
    ----------
    input_path : str
        Path to the raw train_FD001.txt file.

    Returns
    -------
    pd.DataFrame
        Training dataset with meaningful column names.
    """

    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Dataset not found: {input_path}"
        )

    df = pd.read_csv(
        input_path,
        sep=r"\s+",
        header=None,
        names=COLUMNS
    )

    return df


if __name__ == "__main__":

    data_path = "data/raw/CMaps/train_FD001.txt"

    df = load_training_data(data_path)

    print("Data loaded successfully.")
    print(f"Shape: {df.shape}")
    print(f"Number of engines: {df['engine'].nunique()}")
    print("\nFirst 5 rows:")
    print(df.head())