import sys

sys.path.append("src/data")

from data_ingestion import load_training_data
from preprocessing import add_rul_column
from data_split import split_by_engine


DATA_PATH = "data/raw/CMaps/train_FD001.txt"


# Load raw data
df = load_training_data(DATA_PATH)

# Add RUL
df = add_rul_column(df)

# Split by engine
train_df, validation_df = split_by_engine(df)


print("Data split test successful.")

print("\nTraining:")
print(f"Rows: {len(train_df)}")
print(f"Engines: {train_df['engine'].nunique()}")

print("\nValidation:")
print(f"Rows: {len(validation_df)}")
print(f"Engines: {validation_df['engine'].nunique()}")


# Check for engine overlap
train_engines = set(train_df["engine"].unique())
validation_engines = set(validation_df["engine"].unique())

overlap = train_engines.intersection(
    validation_engines
)

print("\nEngine overlap:")
print(overlap)

print("\nTraining RUL range:")
print(
    train_df["RUL"].min(),
    "to",
    train_df["RUL"].max()
)

print("\nValidation RUL range:")
print(
    validation_df["RUL"].min(),
    "to",
    validation_df["RUL"].max()
)