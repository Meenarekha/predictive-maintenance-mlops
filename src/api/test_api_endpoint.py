import json
import urllib.request


API_URL = "http://127.0.0.1:8000/predict"


def test_predict_endpoint():
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

    request_body = json.dumps(request_data).encode("utf-8")

    request = urllib.request.Request(
        API_URL,
        data=request_body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request) as response:
        assert response.status == 200

        result = json.loads(
            response.read().decode("utf-8")
        )

    assert "predicted_RUL" in result
    assert isinstance(result["predicted_RUL"], float)