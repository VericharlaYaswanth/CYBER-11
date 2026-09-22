import pandas as pd

FILE = "data/vulnerability_dataset.csv"

df = pd.read_csv(FILE)

print("Dataset shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

print("\nCVSS statistics:")
print(df["cvss_score"].describe())

print("\nEPSS statistics:")
print(df["epss"].describe())

print("\nCVSS version distribution:")
print(df["cvss_version"].value_counts())

print("\nSnapshot distribution:")
print(df["snapshot_date"].value_counts().sort_index())