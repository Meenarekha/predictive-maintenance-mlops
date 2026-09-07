import json
import urllib.request

import pandas as pd


DATA_PATH = "data/raw/CMaps/train_FD001.txt"
API_URL = "http://127.0.0.1:8000/predict"

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

FEATURE_COLUMNS = [
    "cycle",
    "setting_1",
    "setting_2",
    "lpc_outlet_temperature",
    "hpc_outlet_temperature",
    "lpt_outlet_temperature",
    "bypass_duct_pressure",
    "hpc_outlet_pressure",
    "physical_fan_speed",
    "physical_core_speed",
    "hpc_outlet_static_pressure",
    "fuel_flow_to_ps30_ratio",
    "corrected_fan_speed",
    "corrected_core_speed",
    "bypass_ratio",
    "bleed_enthalpy",
    "high_pressure_turbine_cool_air_flow",
    "low_pressure_turbine_cool_air_flow",
]


# Load the real C-MAPSS dataset
df = pd.read_csv(
    DATA_PATH,
    sep=r"\s+",
    header=None,
    names=COLUMNS,
)

# Calculate the actual RUL
df["max_cycle"] = df.groupby("engine")["cycle"].transform("max")
df["RUL"] = df["max_cycle"] - df["cycle"]

# Select one real data row
row = df[
    (df["engine"] == 1) &
    (df["cycle"] == 100)
].iloc[0]

# Create API request using the same 18 features
request_data = {
    feature: float(row[feature])
    for feature in FEATURE_COLUMNS
}

# Send request to FastAPI
request_body = json.dumps(request_data).encode("utf-8")

request = urllib.request.Request(
    API_URL,
    data=request_body,
    headers={"Content-Type": "application/json"},
    method="POST",
)

with urllib.request.urlopen(request) as response:
    result = json.loads(response.read().decode("utf-8"))

predicted_rul = result["predicted_RUL"]
actual_rul = float(row["RUL"])
absolute_error = abs(actual_rul - predicted_rul)

print("=" * 60)
print("REAL C-MAPSS API TEST")
print("=" * 60)
print(f"Engine:          {int(row['engine'])}")
print(f"Cycle:           {int(row['cycle'])}")
print(f"Actual RUL:      {actual_rul:.2f}")
print(f"Predicted RUL:   {predicted_rul:.2f}")
print(f"Absolute Error:  {absolute_error:.2f}")
print("=" * 60)
print("API test completed successfully.")