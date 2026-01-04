
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from preprocessing import build_preprocessor

df = pd.read_csv("data/raw/heart.csv")
df["target"] = (df["target"] > 0).astype(int)

X = df.drop("target", axis=1)
y = df["target"]

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(n_estimators=200)
}

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        pipeline = Pipeline([
            ("prep", build_preprocessor()),
            ("model", model)
        ])
        pipeline.fit(X, y)
        preds = pipeline.predict(X)

        mlflow.log_metric("accuracy", accuracy_score(y, preds))
        mlflow.log_metric("roc_auc", roc_auc_score(y, preds))
        mlflow.sklearn.log_model(pipeline, "model")
