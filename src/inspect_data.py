import pandas as pd

FILE = "data/epss_combined.csv"

df = pd.read_csv(FILE)

print("\nDataset shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

print("\nData types:")
print(df.dtypes)

print("\nFirst 5 records:")
print(df.head())

print("\nBasic statistics:")
print(df.describe())