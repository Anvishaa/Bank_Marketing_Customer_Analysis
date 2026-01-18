import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


df = pd.read_csv("data/processed/bank_clean.csv")

X = df.drop("y", axis=1)
y = df["y"]


# FEATURES (comment one or two at once)

numeric_features = [
    "age",
    "duration",   # intentionally removed
    "campaign",
    # "pdays",
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
    # "month",
    "day_of_week",
    "poutcome"
]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

# model
model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)
# pipeline
clf = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("model", model)
    ]
)

# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# train & test
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("MODEL WITHOUT 'xyz' ")
print(classification_report(y_test, y_pred))
