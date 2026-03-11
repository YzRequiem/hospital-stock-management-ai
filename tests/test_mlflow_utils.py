"""Tests for MLflow utility helpers."""

from pathlib import Path

import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mlflow_utils import (
    _build_dataset_input_frame,
    _get_dataset_display_name,
    _get_dataset_source,
)


class TestDatasetHelpers:
    """Tests for MLflow dataset helper functions."""

    def test_dataset_display_name_prefers_filename(self):
        """The dataset display name should use the actual CSV filename when available."""
        dataset_path = Path("data/dataset_stock_hopital_ENRICHI.csv")
        assert _get_dataset_display_name("enriched", dataset_path) == "dataset_stock_hopital_ENRICHI.csv"

    def test_dataset_source_contains_filename(self):
        """The dataset source should preserve the underlying file name."""
        dataset_path = Path("data/dataset_stock_hopital_ENRICHI.csv")
        assert _get_dataset_source(dataset_path).endswith("dataset_stock_hopital_ENRICHI.csv")

    def test_build_dataset_input_frame_combines_train_and_test(self):
        """The logged dataset frame should contain both train and test rows."""
        train_df = pd.DataFrame({"ds": pd.date_range("2024-01-01", periods=2), "y": [1.0, 2.0]})
        test_df = pd.DataFrame({"ds": pd.date_range("2024-01-03", periods=1), "y": [3.0]})

        combined = _build_dataset_input_frame(train_df, test_df)

        assert combined is not None
        assert len(combined) == 3
        assert list(combined.columns) == ["ds", "y"]