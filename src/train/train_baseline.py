import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


df = pd.read_csv("data/processed/bank_clean.csv")

X = df.drop("y", axis=1)
y = df["y"]

numeric_features = [
    "age",
    "duration",
    "campaign",
    "pdays",
    "previous",
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed"
]

# everything else is categorical
categorical_features = [
    col for col in X.columns if col not in numeric_features
]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

# model
model = LogisticRegression(
    max_iter=1500,
    class_weight="balanced"
)

# Pipeline

clf = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("model", model)
    ]
)

# train test split 
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.1,
    random_state=42,
    stratify=y
)

# Train
clf.fit(X_train, y_train)

# test
y_pred = clf.predict(X_test)

# Metrics
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
