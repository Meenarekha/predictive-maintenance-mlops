import sys

import numpy as np
import pandas as pd

sys.path.append("src/monitoring")

from drift_detection import (
    calculate_psi,
    classify_drift,
    monitor_features,
)


def test_no_drift():
    reference = pd.Series(
        np.random.default_rng(42).normal(100, 10, 1000)
    )

    current = reference.copy()

    psi = calculate_psi(reference, current)

    assert psi < 0.10
    assert classify_drift(psi) == "No significant drift"


def test_significant_drift():
    rng = np.random.default_rng(42)

    reference = pd.Series(
        rng.normal(100, 10, 1000)
    )

    current = pd.Series(
        rng.normal(160, 10, 1000)
    )

    psi = calculate_psi(reference, current)

    assert psi >= 0.25
    assert classify_drift(psi) == "Significant drift"


def test_monitor_features():
    rng = np.random.default_rng(42)

    reference = pd.DataFrame({
        "cycle": rng.normal(100, 10, 1000),
        "physical_core_speed": rng.normal(8000, 100, 1000),
    })

    current = pd.DataFrame({
        "cycle": rng.normal(100, 10, 1000),
        "physical_core_speed": rng.normal(8000, 100, 1000),
    })

    features = [
        "cycle",
        "physical_core_speed",
    ]

    report = monitor_features(
        reference,
        current,
        features,
    )

    assert len(report) == 2
    assert list(report.columns) == [
        "feature",
        "psi",
        "drift_status",
    ]

    assert set(report["feature"]) == set(features)