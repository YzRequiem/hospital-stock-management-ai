"""
MLflow Tracking Utilities
=========================

Helpers to log prediction experiments without coupling the CLI to MLflow internals.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    mlflow = None
    MLFLOW_AVAILABLE = False


def check_mlflow_available() -> bool:
    """Return True when MLflow can be imported."""
    return MLFLOW_AVAILABLE


def get_default_tracking_uri(project_root: Optional[Path] = None) -> str:
    """Return the default MLflow tracking backend URI based on SQLite."""
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent

    db_path = project_root.resolve() / "mlflow.db"
    return f"sqlite:///{db_path.as_posix()}"


def _serialize_param(value: Any) -> str:
    """Convert values to MLflow-compatible parameter strings."""
    if isinstance(value, Path):
        value = str(value)
    elif isinstance(value, (list, tuple, set)):
        value = ", ".join(str(item) for item in value)
    elif isinstance(value, dict):
        value = json.dumps(value, ensure_ascii=True, sort_keys=True)

    return str(value)[:500]


def _log_params(params: Dict[str, Any]) -> None:
    """Log non-null parameters with safe serialization."""
    for key, value in params.items():
        if value is None:
            continue
        mlflow.log_param(key, _serialize_param(value))


def log_prediction_run(
    product_name: str,
    dataset_name: str,
    horizon_days: int,
    dataset_path: Path,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    test_predictions: pd.DataFrame,
    future_predictions: pd.DataFrame,
    metrics: Dict[str, float],
    prophet_settings: Dict[str, Any],
    model_details: Dict[str, Any],
    experiment_name: str,
    tracking_uri: Optional[str] = None,
    saved_results_dir: Optional[Path] = None,
) -> bool:
    """Log a prediction run, metrics, and generated artifacts to MLflow."""
    if not MLFLOW_AVAILABLE:
        return False

    try:
        mlflow.set_tracking_uri(tracking_uri or get_default_tracking_uri())

        mlflow.set_experiment(experiment_name)

        run_name = f"predict-{product_name.lower().replace(' ', '-')}-{dataset_name}"

        with mlflow.start_run(run_name=run_name):
            mlflow.set_tags({
                "project": "hospital-stock-management-ai",
                "command": "predict",
                "model_type": "Prophet",
                "dataset": dataset_name,
            })

            _log_params({
                "product_name": product_name,
                "dataset_name": dataset_name,
                "horizon_days": horizon_days,
                "dataset_path": dataset_path,
                "train_rows": len(train_df),
                "test_rows": len(test_df),
                **{f"prophet_{key}": value for key, value in prophet_settings.items()},
            })

            mlflow.log_metrics({key: float(value) for key, value in metrics.items()})
            mlflow.log_dict(model_details, "model_summary.json")
            mlflow.log_dict(
                {
                    "product_name": product_name,
                    "dataset_name": dataset_name,
                    "horizon_days": horizon_days,
                    "dataset_path": str(dataset_path),
                },
                "run_context.json",
            )

            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                future_predictions.to_csv(temp_path / "future_predictions.csv", index=False)
                test_predictions.to_csv(temp_path / "test_predictions.csv", index=False)
                (temp_path / "metrics.json").write_text(
                    json.dumps(metrics, indent=2),
                    encoding="utf-8",
                )
                mlflow.log_artifacts(str(temp_path), artifact_path="outputs")

            if saved_results_dir and saved_results_dir.exists():
                for artifact in saved_results_dir.iterdir():
                    if artifact.is_file():
                        mlflow.log_artifact(str(artifact), artifact_path="saved_results")

        return True

    except Exception as exc:
        print(f"⚠️ MLflow tracking ignoré: {exc}")
        return False