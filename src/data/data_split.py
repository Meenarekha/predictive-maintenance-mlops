import pandas as pd
from sklearn.model_selection import train_test_split


def split_by_engine(
    df: pd.DataFrame,
    validation_size: float = 0.2,
    random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset by engine ID.

    This prevents rows belonging to the same engine
    from appearing in both training and validation data.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing an 'engine' column.

    validation_size : float
        Fraction of engines assigned to validation.

    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Training dataframe and validation dataframe.
    """

    engine_ids = df["engine"].unique()

    train_engines, validation_engines = train_test_split(
        engine_ids,
        test_size=validation_size,
        random_state=random_state
    )

    train_df = df[
        df["engine"].isin(train_engines)
    ].copy()

    validation_df = df[
        df["engine"].isin(validation_engines)
    ].copy()

    return train_df, validation_df