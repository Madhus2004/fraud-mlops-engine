import os
from pathlib import Path


def log_run(run_name: str, params: dict, metrics: dict, files=()):
    """MLflow logging that degrades to a no-op if mlflow is not installed."""
    try:
        import mlflow
    except ImportError:
        print("mlflow not installed; skipping experiment tracking.")
        return
    try:  # tracking must never break training or promotion
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
        try:
            mlflow.set_experiment_tag("mlflow.experimentKind", "custom_model_development")
        except Exception:
            pass
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({k: str(v) for k, v in params.items()})
            mlflow.log_metrics({k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))})
            for f in files:
                if Path(f).exists():
                    mlflow.log_artifact(str(f))
    except Exception as exc:
        print(f"MLflow logging skipped: {exc}")
