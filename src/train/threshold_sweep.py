import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, confusion_matrix

df = pd.read_csv("data/processed/bank_clean.csv")

X = df.drop("y", axis=1)
y = df["y"]

numeric_features = [
    "age",
    "campaign",
    "pdays",
    "previous",
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed"
]

categorical_features = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "poutcome"
]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

clf = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("model", model)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

clf.fit(X_train, y_train)

y_proba = clf.predict_proba(X_test)[:, 1]

# THRESHOLD SWEEP

thresholds = [0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7]

print("threshold | precision | recall | predicted_positives")
print("-" * 55)

for t in thresholds:
    y_pred = (y_proba >= t).astype(int)

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred)
    positives = y_pred.sum()

    print(f"{t:8.2f} | {precision:9.2f} | {recall:6.2f} | {positives:20d}")
