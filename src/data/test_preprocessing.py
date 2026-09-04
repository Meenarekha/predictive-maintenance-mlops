from data_ingestion import load_training_data
from preprocessing import add_rul_column


DATA_PATH = "data/raw/CMaps/train_FD001.txt"


df = load_training_data(DATA_PATH)

df = add_rul_column(df)

print("Preprocessing test successful.")
print(f"Shape: {df.shape}")

print("\nRUL statistics:")
print(df["RUL"].describe())

print("\nFirst 5 rows:")
print(
    df[["engine", "cycle", "RUL"]].head()
)

print("\nLast row of engine 1:")
print(
    df[df["engine"] == 1][
        ["engine", "cycle", "RUL"]
    ].tail(1)
)