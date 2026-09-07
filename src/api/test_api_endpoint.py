import sys

sys.path.append("src/api")

from fastapi.testclient import TestClient
from sklearn.dummy import DummyRegressor

import main


def test_predict_endpoint(monkeypatch):
    # Create a simple test model
    test_model = DummyRegressor(strategy="constant", constant=100.0)

    # Train it with the expected 18 features
    test_model.fit(
        [[0] * 18],
        [100.0]
    )

    # Replace the production model with the test model
    monkeypatch.setattr(
        main,
        "model",
        test_model
    )

    client = TestClient(main.app)

    request_data = {
        "cycle": 100,
        "setting_1": 0.0023,
        "setting_2": -0.0003,
        "lpc_outlet_temperature": 642.0,
        "hpc_outlet_temperature": 1589.0,
        "lpt_outlet_temperature": 1410.0,
        "bypass_duct_pressure": 552.0,
        "hpc_outlet_pressure": 2388.0,
        "physical_fan_speed": 9050.0,
        "physical_core_speed": 2388.0,
        "hpc_outlet_static_pressure": 47.0,
        "fuel_flow_to_ps30_ratio": 522.0,
        "corrected_fan_speed": 2388.0,
        "corrected_core_speed": 8140.0,
        "bypass_ratio": 8.4,
        "bleed_enthalpy": 392.0,
        "high_pressure_turbine_cool_air_flow": 1.3,
        "low_pressure_turbine_cool_air_flow": 1.0,
    }

    response = client.post(
        "/predict",
        json=request_data
    )

    assert response.status_code == 200

    result = response.json()

    assert "predicted_RUL" in result
    assert isinstance(result["predicted_RUL"], float)
    assert result["predicted_RUL"] == 100.0