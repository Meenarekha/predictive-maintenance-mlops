import sys

sys.path.append("src/data")
sys.path.append("src/features")

from data_ingestion import load_training_data
from preprocessing import add_rul_column
from feature_selection import select_features


DATA_PATH = "data/raw/CMaps/train_FD001.txt"


# Load data
df = load_training_data(DATA_PATH)

# Add target
df = add_rul_column(df)

# Select model features
X, feature_names = select_features(df)


print("Feature selection test successful.")

print("\nOriginal dataframe shape:")
print(df.shape)

print("\nModel input shape:")
print(X.shape)

print("\nNumber of selected features:")
print(len(feature_names))

print("\nSelected features:")

for i, feature in enumerate(feature_names, 1):
    print(f"{i:2}. {feature}")