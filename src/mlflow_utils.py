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


def normalize_tracking_uri(
    tracking_uri: Optional[str],
    project_root: Optional[Path] = None,
) -> str:
    """Normalize user-provided MLflow tracking URIs and fallback placeholders."""
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent

    if tracking_uri is None:
        return get_default_tracking_uri(project_root)

    cleaned = tracking_uri.strip()
    if cleaned.lower() in {"", "string", "none", "null", "undefined"}:
        return get_default_tracking_uri(project_root)

    sqlite_prefix = "sqlite:///"
    if cleaned.startswith(sqlite_prefix):
        sqlite_path = cleaned[len(sqlite_prefix):]
        is_windows_absolute = len(sqlite_path) > 1 and sqlite_path[1] == ":"
        if sqlite_path and not Path(sqlite_path).is_absolute() and not is_windows_absolute:
            resolved_db_path = (project_root / sqlite_path).resolve()
            return f"sqlite:///{resolved_db_path.as_posix()}"

    if "://" not in cleaned:
        return str((project_root / cleaned).resolve())

    return cleaned


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


def _serialize_tag(value: Any) -> str:
    """Convert values to MLflow-compatible tag strings."""
    if isinstance(value, bool):
        return str(value).lower()
    return _serialize_param(value)


def _log_tags(tags: Dict[str, Any]) -> None:
    """Log non-null tags with safe serialization."""
    for key, value in tags.items():
        if value is None:
            continue
        mlflow.set_tag(key, _serialize_tag(value))


def _build_model_input_example(train_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Build a small input example compatible with Prophet inference."""
    if not isinstance(train_df, pd.DataFrame) or "ds" not in train_df.columns:
        return None

    input_columns = ["ds"] + [
        column for column in train_df.columns if column not in {"ds", "y"}
    ]
    return train_df[input_columns].head(5).copy()


def _get_dataset_display_name(dataset_name: str, dataset_path: Path) -> str:
    """Build a readable dataset name for MLflow inputs."""
    if isinstance(dataset_path, Path) and dataset_path.name:
        return dataset_path.name
    return dataset_name


def _get_dataset_source(dataset_path: Path) -> str:
    """Return a stable dataset source string for MLflow dataset tracking."""
    resolved_path = dataset_path.resolve()
    try:
        return resolved_path.as_uri()
    except ValueError:
        return resolved_path.as_posix()


def _build_dataset_input_frame(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Optional[pd.DataFrame]:
    """Combine the modeling data into a single MLflow dataset input."""
    frames = []
    for frame in (train_df, test_df):
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            frames.append(frame.copy())

    if not frames:
        return None

    return pd.concat(frames, ignore_index=True)


def _log_input_dataset(
    dataset_name: str,
    dataset_path: Path,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """Log the input dataset using MLflow's native dataset tracking APIs."""
    dataset_frame = _build_dataset_input_frame(train_df, test_df)
    if dataset_frame is None:
        return

    if not hasattr(mlflow, "data") or not hasattr(mlflow, "log_input"):
        return

    dataset_kwargs: Dict[str, Any] = {
        "source": _get_dataset_source(dataset_path),
        "name": _get_dataset_display_name(dataset_name, dataset_path),
    }
    if "y" in dataset_frame.columns:
        dataset_kwargs["targets"] = "y"

    dataset = mlflow.data.from_pandas(dataset_frame, **dataset_kwargs)
    mlflow.log_input(dataset, context="training")


def _ensure_experiment_ready(experiment_name: str) -> None:
    """Restore a deleted experiment before activating it when possible."""
    try:
        from mlflow.entities import ViewType
        from mlflow.tracking import MlflowClient

        client = MlflowClient()
        experiments = client.search_experiments(
            view_type=ViewType.ALL,
            filter_string=f"name = '{experiment_name}'",
            max_results=100,
        )

        for experiment in experiments:
            if experiment.name != experiment_name:
                continue
            if experiment.lifecycle_stage == "deleted":
                client.restore_experiment(experiment.experiment_id)
            break
    except Exception:
        # Fallback to the default MLflow behavior if experiment inspection/restoration fails.
        pass


def _set_experiment_tags(experiment_name: str, tags: Dict[str, Any]) -> None:
    """Persist experiment-level tags for easier filtering in the MLflow UI."""
    try:
        from mlflow.tracking import MlflowClient

        client = MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            return

        for key, value in tags.items():
            if value is None:
                continue
            client.set_experiment_tag(
                experiment.experiment_id,
                key,
                _serialize_tag(value),
            )
    except Exception:
        # Experiment tags are useful metadata, but should not block run logging.
        pass


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
    model: Optional[Any] = None,
    run_tags: Optional[Dict[str, Any]] = None,
) -> bool:
    """Log a prediction run, metrics, and generated artifacts to MLflow."""
    if not MLFLOW_AVAILABLE:
        return False

    try:
        mlflow.set_tracking_uri(normalize_tracking_uri(tracking_uri))

        _ensure_experiment_ready(experiment_name)
        mlflow.set_experiment(experiment_name)

        _set_experiment_tags(
            experiment_name,
            {
                "project": "hospital-stock-management-ai",
                "model_type": "Prophet",
                "model_flavor": "mlflow.prophet",
                "dataset": dataset_name,
                "source": (run_tags or {}).get("source"),
                "analysis_type": (run_tags or {}).get("analysis_type"),
                "tracking_backend": normalize_tracking_uri(tracking_uri),
            },
        )

        run_name = f"predict-{product_name.lower().replace(' ', '-')}-{dataset_name}"

        with mlflow.start_run(run_name=run_name):
            _log_tags({
                "project": "hospital-stock-management-ai",
                "command": "predict",
                "model_type": "Prophet",
                "dataset": dataset_name,
                "dataset_display_name": _get_dataset_display_name(dataset_name, dataset_path),
                "dataset_source": _get_dataset_source(dataset_path),
                "product_name": product_name,
                "horizon_days": horizon_days,
                "dataset_path": dataset_path,
                "has_regressors": bool(model_details.get("regressors")),
                "regressor_count": len(model_details.get("regressors", [])),
                "results_dir_present": bool(saved_results_dir and saved_results_dir.exists()),
                **(run_tags or {}),
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

            try:
                _log_input_dataset(
                    dataset_name=dataset_name,
                    dataset_path=dataset_path,
                    train_df=train_df,
                    test_df=test_df,
                )
                mlflow.set_tag("input_dataset_logged", "true")
            except Exception as dataset_exc:
                mlflow.set_tag("input_dataset_logged", "false")
                mlflow.set_tag("dataset_logging_error", str(dataset_exc)[:500])

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
                mlflow.log_artifacts(str(saved_results_dir), artifact_path="saved_results")

            if model is not None:
                try:
                    import mlflow.prophet as mlflow_prophet

                    mlflow_prophet.log_model(
                        pr_model=model,
                        name="prophet",
                        input_example=_build_model_input_example(train_df),
                    )
                    mlflow.set_tag("model_logged", "true")
                except Exception as model_exc:
                    mlflow.set_tag("model_logged", "false")
                    mlflow.set_tag("model_logging_error", str(model_exc)[:500])

        return True

    except Exception as exc:
        print(f"⚠️ MLflow tracking ignoré: {exc}")
        return False