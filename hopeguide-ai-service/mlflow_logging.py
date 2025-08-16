import mlflow
from pathlib import Path

# ====== CONFIG ======
MLFLOW_TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME = "Hopeguide-Checkpoints"
CHECKPOINT_PATH = Path("hopeguide-ai-service/openchat-phq9-lora/checkpoint-1880")
# ====================

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(f"Checkpoint path does not exist: {CHECKPOINT_PATH}")

with mlflow.start_run(run_name="hopeguide_checkpoint"):
    mlflow.log_artifacts(str(CHECKPOINT_PATH), artifact_path="hopeguide_model")
    print(f"✅ Logged checkpoint from {CHECKPOINT_PATH} to MLflow")
