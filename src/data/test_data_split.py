from data_ingestion import load_training_data
from preprocessing import add_rul_column
from data_split import split_by_engine


DATA_PATH = "data/raw/CMaps/train_FD001.txt"


def test_split_by_engine():
    # Load data
    df = load_training_data(DATA_PATH)

    # Add RUL
    df = add_rul_column(df)

    # Split by engine
    train_df, validation_df = split_by_engine(df)

    # Check total rows are preserved
    assert len(train_df) + len(validation_df) == len(df)

    # Check expected number of engines
    assert train_df["engine"].nunique() == 80
    assert validation_df["engine"].nunique() == 20

    # Check there is no engine overlap
    train_engines = set(train_df["engine"].unique())
    validation_engines = set(validation_df["engine"].unique())

    assert train_engines.isdisjoint(validation_engines)

    # Check RUL exists
    assert "RUL" in train_df.columns
    assert "RUL" in validation_df.columns

    # Check expected RUL ranges
    assert train_df["RUL"].min() == 0
    assert validation_df["RUL"].min() == 0