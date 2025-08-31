import argparse, mlflow, mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--n_estimators", type=int, default=200)
parser.add_argument("--max_depth", type=int, default=5)
args = parser.parse_args()

X, y = load_iris(return_X_y=True, as_frame=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.20, random_state=42, stratify=y)

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("iris-model-training")

with mlflow.start_run() as run:
    mlflow.log_params(vars(args))

    model = RandomForestClassifier(
        n_estimators = args.n_estimators,
        max_depth=args.max_depth,
        random_state=42
    ).fit(Xtr, ytr)

    yhat = model.predict(Xte)
    acc = accuracy_score(yte, yhat)
    f1 = f1_score(yte, yhat, average="macro")

    mlflow.log_metrics({ 'accuracy': acc, 'f1_macro': f1 })

    mlflow.sklearn.log_model(
        sk_model=model,
        name='model',
        signature=mlflow.models.infer_signature(Xtr, model.predict(Xtr)),
        input_example=Xtr.head(2)
    )