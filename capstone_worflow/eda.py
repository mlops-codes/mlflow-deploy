from pathlib import Path
import json
import matplotlib.pyplot as plt
import mlflow
from config import settings
from data import load_dataset

def run_eda():
    df = load_dataset()
    out = Path(settings.artifacts_dir) / "eda"
    out.mkdir(parents=True, exist_ok=True)

    summary = {
        "rows": int(df.shape[0]),
        "columns": list(df.columns),
        "describe": json.loads(df.describe().to_json()),
        "correlation": df.corr(numeric_only=True).to_dict(),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    plt.figure()
    plt.scatter(df["YearsExperience"], df["Salary"])
    plt.xlabel("YearsExperience")
    plt.ylabel("Salary")
    plt.title("Salary vs YearsExperience")
    scatter_path = out / "scatter.png"
    plt.savefig(scatter_path, bbox_inches="tight")
    plt.close()

    plt.figure()
    df["Salary"].plot(kind="hist", bins=10)
    plt.xlabel("Salary")
    plt.title("Salary Distribution")
    hist_path = out / "salary_hist.png"
    plt.savefig(hist_path, bbox_inches="tight")
    plt.close()

    mlflow.set_tracking_uri(settings.tracking_uri)
    mlflow.set_experiment(settings.experiment_name)

    with mlflow.start_run(run_name="eda") as run:
        mlflow.log_artifact(str(out / "summary.json"), artifact_path="eda")
        mlflow.log_artifact(str(scatter_path), artifact_path="eda")
        mlflow.log_artifact(str(hist_path), artifact_path="eda")

        print("=== EDA completed ===")
        print(f"Run ID: {run.info.run_id}")
        print(f"Experiment: {settings.experiment_name}")
        print(f"Artifacts logged to: {mlflow.get_artifact_uri()}")

if __name__ == "__main__":
    run_eda()
