import mlflow
from mlflow.tracking import MlflowClient

EXPERIMENT = "iris-model-training"
MODEL_NAME = "iris_rf_classifier"
ALIAS = "prod"

mlflow.set_tracking_uri("http://127.0.0.1:5000")
client = MlflowClient()

# 1. Get experiment
exp = client.get_experiment_by_name(EXPERIMENT)
if exp is None:
    raise ValueError(f"Experiment '{EXPERIMENT}' not found!")

# 2. Pick the best run (highest f1_macro)
runs = client.search_runs(
    [exp.experiment_id],
    order_by=["metrics.f1_macro DESC"],
    max_results=1,
)
if not runs:
    raise ValueError(f"No runs found in experiment '{EXPERIMENT}'.")

best = runs[0]
run_id = best.info.run_id
print(f"Best run: {run_id} (f1_macro={best.data.metrics.get('f1_macro')})")

# 3. Register a new version of the model
model_uri = f"runs:/{run_id}/model"
mv = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)
print(f"Created new version {mv.version} of model '{MODEL_NAME}'")

# 4. Update alias (e.g. 'prod' → always points to latest best model)
client.set_registered_model_alias(
    name=MODEL_NAME,
    alias=ALIAS,
    version=mv.version,
)
print(f"Alias '{ALIAS}' now points to {MODEL_NAME} v{mv.version}")
