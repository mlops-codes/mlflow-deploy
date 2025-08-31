from pathlib import Path
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import settings
from data import load_dataset
from features import split_xy
from model import linear, decision_tree, random_forest

def eval_split(model, Xs, ys, prefix):
    preds = model.predict(Xs)
    mae = mean_absolute_error(ys, preds)
    rmse = mean_squared_error(ys, preds, squared=False)
    r2 = r2_score(ys, preds)
    mlflow.log_metrics({f"{prefix}_mae": mae, f"{prefix}_rmse": rmse, f"{prefix}_r2": r2})
    return {"mae": mae, "rmse": rmse, "r2": r2}

def train_one(name, model, X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name=f"train-{name}") as run:
        mlflow.log_params({"model_name": name, "test_size": settings.test_size, "random_state": settings.random_state})
        if hasattr(model, "get_params"):
            p = model.get_params()
            keep = {"n_estimators", "max_depth", "min_samples_split", "min_samples_leaf", "random_state"}
            small = {k: v for k, v in p.items() if k in keep}
            if small: mlflow.log_params({f"hp_{k}": v for k, v in small.items()})
        model.fit(X_train, y_train)
        _ = eval_split(model, X_train, y_train, "train")
        test_metrics = eval_split(model, X_test, y_test, "test")
        sig = infer_signature(X_train, model.predict(X_train))
        logged = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=f"model_{name}",
            signature=sig,
            input_example=X_train.head(3),
            registered_model_name=(settings.registered_name if settings.register else None),
        )
        return {"name": name, "run_id": run.info.run_id, "model_uri": logged.model_uri, "metrics": test_metrics}

def main():
    df = load_dataset()
    X, y = split_xy(df)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=settings.test_size, random_state=settings.random_state)

    mlflow.set_tracking_uri(settings.tracking_uri)
    mlflow.set_experiment(settings.experiment_name)

    candidates = {
        "linear": linear.build(),
        "decision_tree": decision_tree.build(random_state=settings.random_state),
        "random_forest": random_forest.build(n_estimators=200, random_state=settings.random_state),
    }

    results = [train_one(n, m, X_tr, X_te, y_tr, y_te) for n, m in candidates.items()]
    best = sorted(results, key=lambda r: r["metrics"]["rmse"])[0]

    with mlflow.start_run(run_name=f"select-best-{best['name']}"):
        mlflow.log_params({"selected_model": best["name"]})
        best_model = mlflow.sklearn.load_model(best["model_uri"])
        out_dir = Path(settings.artifacts_dir) / "model"
        out_dir.parent.mkdir(parents=True, exist_ok=True)
        mlflow.sklearn.save_model(best_model, path=str(out_dir))
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model_best",
            signature=infer_signature(X_tr, best_model.predict(X_tr)),
            input_example=X_tr.head(3),
            registered_model_name=(settings.registered_name if settings.register else None),
        )
        print("BEST:", best)

if __name__ == "__main__":
    main()
