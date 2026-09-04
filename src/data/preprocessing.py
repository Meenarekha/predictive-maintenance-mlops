import pandas as pd


def add_rul_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Remaining Useful Life (RUL) for each engine.

    RUL is calculated as:

        RUL = maximum cycle of engine - current cycle

    Parameters
    ----------
    df : pd.DataFrame
        Raw C-MAPSS training data.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the RUL column.
    """

    df = df.copy()

    # Find the final cycle reached by each engine
    max_cycle = (
        df.groupby("engine")["cycle"]
        .transform("max")
    )

    # Calculate Remaining Useful Life
    df["RUL"] = max_cycle - df["cycle"]

    return df


def remove_constant_features(
    df: pd.DataFrame,
    constant_features: list[str]
) -> pd.DataFrame:
    """
    Remove features that have the same value for every row.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    constant_features : list[str]
        Columns identified as constant during EDA.

    Returns
    -------
    pd.DataFrame
        DataFrame with constant features removed.
    """

    df = df.copy()

    existing_features = [
        feature
        for feature in constant_features
        if feature in df.columns
    ]

    df = df.drop(
        columns=existing_features
    )

    return df