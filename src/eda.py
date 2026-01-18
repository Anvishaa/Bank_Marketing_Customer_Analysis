import pandas as pd

df = pd.read_csv("data/processed/bank_clean.csv")

# Overall conversion rate
print("Overall conversion rate:")
print(df["y"].value_counts(normalize=True))

# Conversion by contact type
print("\nConversion by contact:")
print(df.groupby("contact")["y"].mean())

# Conversion by month
print("\nConversion by month:")
print(df.groupby("month")["y"].mean())

# Conversion by day 
print(
    df.groupby("day_of_week")["y"].mean().reindex(["mon","tue","wed","thu","fri"])

)

