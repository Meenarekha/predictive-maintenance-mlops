from data_ingestion import load_training_data
from preprocessing import add_rul_column


DATA_PATH = "data/raw/CMaps/train_FD001.txt"


def test_add_rul_column():
    df = load_training_data(DATA_PATH)

    df = add_rul_column(df)

    # Basic checks
    assert "RUL" in df.columns
    assert len(df) == 20631

    # RUL should never be negative
    assert df["RUL"].min() == 0

    # Engine 1 should finish at RUL = 0
    engine_1_last = df[df["engine"] == 1].tail(1).iloc[0]

    assert engine_1_last["cycle"] == 192
    assert engine_1_last["RUL"] == 0