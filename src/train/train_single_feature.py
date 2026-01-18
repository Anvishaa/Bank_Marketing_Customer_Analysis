import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

df = pd.read_csv("data/processed/bank_clean.csv")

# CHOOSE EXACTLY ONE FEATURE

# FEATURE = "duration"
FEATURE = "euribor3m"
# FEATURE = "nr.employed"
# FEATURE = "emp.var.rate"
# FEATURE = "cons.price.idx"
# FEATURE = "cons.conf.idx"
# FEATURE = "campaign"
# FEATURE = "pdays"
# FEATURE = "previous"
# FEATURE = "age"

# CATEGORICAL (must be encoded manually later if used)
# FEATURE = "contact"
# FEATURE = "month"
# FEATURE = "poutcome"


try:
    FEATURE
except NameError:
    raise Exception("Uncomment exactly ONE FEATURE")

X = df[[FEATURE]]
y = df["y"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"SINGLE FEATURE MODEL: {FEATURE}")
print(classification_report(y_test, y_pred))
