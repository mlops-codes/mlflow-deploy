import mlflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
from mlflow.tracking import MlflowClient
client = MlflowClient()

EXPERIMENT = 'iris-model-training'
MODEL_NAME = 'iris_rf_classifier'

client = MlflowClient()
exp = client.get_experiment_by_name(EXPERIMENT)
runs = client.search_runs([exp.experiment_id], order_by=['metrics.f1_macro DESC'], max_results=1)

best = runs[0]
run_id = best.info.run_id

source = f"runs:/{run_id}/model"
mv = mlflow.register_model(model_uri=source, name=MODEL_NAME)

print(run_id, mv.version)
