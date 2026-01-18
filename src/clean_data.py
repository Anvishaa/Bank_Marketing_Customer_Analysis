import pandas as pd

df = pd.read_csv("data/raw/bank-additional-full.csv", sep=";")

# convert target var to binary
df["y"] = df["y"].map({"yes": 1, "no": 0})

# check if missing-like values
for col in df.columns:
    print(col, df[col].isin(["unknown"]).sum())

df = df.replace("unknown", pd.NA)

df = df.dropna()

df.to_csv("data/processed/bank_clean.csv", index=False)

print("Cleaned data saved")

