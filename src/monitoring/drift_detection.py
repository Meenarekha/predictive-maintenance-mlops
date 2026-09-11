import numpy as np
import pandas as pd


def calculate_psi(
    reference: pd.Series,
    current: pd.Series,
    bins: int = 10
) -> float:
    """
    Calculate Population Stability Index (PSI).

    PSI measures how much the distribution of a feature
    has changed between reference and current data.

    Parameters
    ----------
    reference : pd.Series
        Reference/training distribution.

    current : pd.Series
        Current/production distribution.

    bins : int
        Number of bins used to compare distributions.

    Returns
    -------
    float
        PSI value.
    """

    reference = pd.Series(reference).dropna()
    current = pd.Series(current).dropna()

    # Create bins based on reference data
    breakpoints = np.percentile(
        reference,
        np.linspace(0, 100, bins + 1)
    )

    # Remove duplicate boundaries
    breakpoints = np.unique(breakpoints)

    # If the feature has insufficient unique values,
    # there is no meaningful drift calculation.
    if len(breakpoints) < 3:
        return 0.0

    reference_counts = pd.cut(
        reference,
        bins=breakpoints,
        include_lowest=True
    ).value_counts(normalize=True)

    current_counts = pd.cut(
        current,
        bins=breakpoints,
        include_lowest=True
    ).value_counts(normalize=True)

    # Align both distributions
    reference_counts, current_counts = (
        reference_counts.align(
            current_counts,
            fill_value=0
        )
    )

    # Avoid division by zero
    epsilon = 1e-6

    reference_counts = reference_counts.clip(lower=epsilon)
    current_counts = current_counts.clip(lower=epsilon)

    psi = np.sum(
        (current_counts - reference_counts)
        * np.log(current_counts / reference_counts)
    )

    return float(psi)


def classify_drift(psi: float) -> str:
    """
    Classify drift based on PSI.

    PSI < 0.10  -> No significant drift
    PSI < 0.25  -> Moderate drift
    PSI >= 0.25 -> Significant drift
    """

    if psi < 0.10:
        return "No significant drift"
    elif psi < 0.25:
        return "Moderate drift"
    else:
        return "Significant drift"


def monitor_features(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature_names: list[str]
) -> pd.DataFrame:
    """
    Calculate PSI and drift status for model features.
    """

    results = []

    for feature in feature_names:

        if feature not in reference_df.columns:
            continue

        if feature not in current_df.columns:
            continue

        psi = calculate_psi(
            reference_df[feature],
            current_df[feature]
        )

        results.append({
            "feature": feature,
            "psi": round(psi, 6),
            "drift_status": classify_drift(psi)
        })

    return pd.DataFrame(results)