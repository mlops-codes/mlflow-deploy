import os
from dataclasses import dataclass
from dotenv import load_dotenv
load_dotenv(override=True)

@dataclass(frozen=True)
class Settings:
    tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    experiment_name: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "salary-regression")
    data_path: str = os.getenv("DATA_PATH", "data/salary.csv")
    artifacts_dir: str = os.getenv("ARTIFACTS_DIR", "artifacts")
    test_size: float = float(os.getenv("TEST_SIZE", "0.2"))
    random_state: int = int(os.getenv("RANDOM_STATE", "42"))
    register: bool = os.getenv("REGISTER_MODEL", "false").lower() == "true"
    registered_name: str = os.getenv("REGISTERED_MODEL_NAME", "salary-model")

settings = Settings()
    