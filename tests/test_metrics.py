"""
Tests for Metrics Module
========================

Unit tests for prediction metrics calculation functions.
"""

import pytest
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.metrics import (
    calculate_mape,
    calculate_mae,
    calculate_rmse,
    calculate_r2,
    calculate_metrics,
    interpret_mape,
    print_metrics,
    compare_models,
    generate_metrics_report
)


class TestCalculateMAE:
    """Tests for Mean Absolute Error calculation."""
    
    def test_perfect_prediction(self):
        """MAE should be 0 for perfect predictions."""
        y_true = np.array([10, 20, 30, 40, 50])
        y_pred = np.array([10, 20, 30, 40, 50])
        
        mae = calculate_mae(y_true, y_pred)
        assert mae == 0.0
    
    def test_known_mae(self):
        """Test MAE with known values."""
        y_true = np.array([10, 20, 30])
        y_pred = np.array([12, 18, 33])  # errors: 2, 2, 3
        
        mae = calculate_mae(y_true, y_pred)
        expected = (2 + 2 + 3) / 3
        assert abs(mae - expected) < 1e-10
    
    def test_symmetric_errors(self):
        """MAE should treat positive and negative errors equally."""
        y_true = np.array([10, 10])
        y_pred_over = np.array([15, 15])
        y_pred_under = np.array([5, 5])
        
        mae_over = calculate_mae(y_true, y_pred_over)
        mae_under = calculate_mae(y_true, y_pred_under)
        
        assert mae_over == mae_under


class TestCalculateMAPE:
    """Tests for Mean Absolute Percentage Error calculation."""
    
    def test_perfect_prediction(self):
        """MAPE should be 0 for perfect predictions."""
        y_true = np.array([10, 20, 30, 40, 50])
        y_pred = np.array([10, 20, 30, 40, 50])
        
        mape = calculate_mape(y_true, y_pred)
        assert mape == 0.0
    
    def test_known_mape(self):
        """Test MAPE with known values."""
        y_true = np.array([100, 100, 100])
        y_pred = np.array([90, 110, 100])  # 10%, 10%, 0% error
        
        mape = calculate_mape(y_true, y_pred)
        expected = (10 + 10 + 0) / 3  # ~6.67%
        assert abs(mape - expected) < 0.01
    
    def test_exclude_zeros(self):
        """Test that zero values are excluded by default."""
        y_true = np.array([0, 0, 100, 100])
        y_pred = np.array([10, 10, 90, 110])
        
        mape = calculate_mape(y_true, y_pred, exclude_zeros=True)
        # Only considers last two values: 10% error each
        expected = 10.0
        assert abs(mape - expected) < 0.01
    
    def test_include_zeros_with_epsilon(self):
        """Test MAPE calculation when zeros are not excluded."""
        y_true = np.array([100, 100])
        y_pred = np.array([90, 110])
        
        mape = calculate_mape(y_true, y_pred, exclude_zeros=False)
        assert mape > 0
    
    def test_all_zeros_returns_zero(self):
        """Test that all-zero actuals return 0 MAPE."""
        y_true = np.array([0, 0, 0])
        y_pred = np.array([10, 20, 30])
        
        mape = calculate_mape(y_true, y_pred, exclude_zeros=True)
        assert mape == 0.0
    
    def test_mape_scale(self):
        """MAPE should be in percentage (0-100+ range)."""
        y_true = np.array([100, 100, 100])
        y_pred = np.array([50, 50, 50])  # 50% error
        
        mape = calculate_mape(y_true, y_pred)
        assert 45 < mape < 55  # Around 50%


class TestCalculateRMSE:
    """Tests for Root Mean Square Error calculation."""
    
    def test_perfect_prediction(self):
        """RMSE should be 0 for perfect predictions."""
        y_true = np.array([10, 20, 30, 40, 50])
        y_pred = np.array([10, 20, 30, 40, 50])
        
        rmse = calculate_rmse(y_true, y_pred)
        assert rmse == 0.0
    
    def test_known_rmse(self):
        """Test RMSE with known values."""
        y_true = np.array([10, 20, 30])
        y_pred = np.array([13, 17, 27])  # errors: 3, 3, 3
        
        rmse = calculate_rmse(y_true, y_pred)
        expected = 3.0  # sqrt(9) = 3
        assert abs(rmse - expected) < 1e-10
    
    def test_rmse_penalizes_large_errors(self):
        """RMSE should penalize large errors more than MAE."""
        y_true = np.array([100, 100])
        y_pred = np.array([90, 110])  # errors: 10, 10
        
        mae = calculate_mae(y_true, y_pred)
        rmse = calculate_rmse(y_true, y_pred)
        
        # For equal errors, RMSE equals MAE
        assert abs(mae - rmse) < 1e-10
        
        # Now with unequal errors
        y_pred_unequal = np.array([80, 100])  # errors: 20, 0
        mae_unequal = calculate_mae(y_true, y_pred_unequal)
        rmse_unequal = calculate_rmse(y_true, y_pred_unequal)
        
        # RMSE > MAE when errors are unequal
        assert rmse_unequal > mae_unequal


class TestCalculateR2:
    """Tests for R² coefficient calculation."""
    
    def test_perfect_prediction(self):
        """R² should be 1 for perfect predictions."""
        y_true = np.array([10, 20, 30, 40, 50])
        y_pred = np.array([10, 20, 30, 40, 50])
        
        r2 = calculate_r2(y_true, y_pred)
        assert abs(r2 - 1.0) < 1e-10
    
    def test_mean_prediction(self):
        """R² should be 0 for predictions equal to mean."""
        y_true = np.array([10, 20, 30, 40, 50])
        y_pred = np.array([30, 30, 30, 30, 30])  # All mean
        
        r2 = calculate_r2(y_true, y_pred)
        assert abs(r2) < 1e-10
    
    def test_worse_than_mean(self):
        """R² can be negative for predictions worse than mean."""
        y_true = np.array([10, 20, 30])
        y_pred = np.array([100, 100, 100])  # Very bad predictions
        
        r2 = calculate_r2(y_true, y_pred)
        assert r2 < 0
    
    def test_r2_bounds(self):
        """R² should generally be between 0 and 1 for reasonable predictions."""
        np.random.seed(42)
        y_true = np.random.uniform(10, 100, 100)
        y_pred = y_true + np.random.normal(0, 10, 100)
        
        r2 = calculate_r2(y_true, y_pred)
        assert 0 < r2 < 1


class TestCalculateMetrics:
    """Tests for combined metrics calculation."""
    
    def test_all_metrics_present(self, sample_predictions):
        """Test that all metrics are returned."""
        y_true, y_pred = sample_predictions
        
        metrics = calculate_metrics(y_true, y_pred)
        
        assert 'mae' in metrics
        assert 'mape' in metrics
        assert 'rmse' in metrics
        assert 'r2' in metrics
    
    def test_metrics_types(self, sample_predictions):
        """Test that all metrics are floats."""
        y_true, y_pred = sample_predictions
        
        metrics = calculate_metrics(y_true, y_pred)
        
        for key, value in metrics.items():
            assert isinstance(value, float), f"{key} is not a float"
    
    def test_metrics_reasonable_ranges(self, sample_predictions):
        """Test that metrics are in reasonable ranges."""
        y_true, y_pred = sample_predictions
        
        metrics = calculate_metrics(y_true, y_pred)
        
        assert metrics['mae'] >= 0
        assert metrics['mape'] >= 0
        assert metrics['rmse'] >= 0
        # R² can be negative but usually positive for reasonable predictions


class TestInterpretMAPE:
    """Tests for MAPE interpretation."""
    
    @pytest.mark.parametrize("mape,expected_level", [
        (5, "excellent"),
        (12, "very_good"),
        (20, "good"),
        (40, "acceptable"),
        (60, "poor"),
    ])
    def test_quality_levels(self, mape, expected_level):
        """Test MAPE quality level interpretation."""
        level, description = interpret_mape(mape)
        assert level == expected_level
    
    def test_description_not_empty(self):
        """Test that description is always returned."""
        for mape in [5, 12, 20, 40, 60]:
            level, description = interpret_mape(mape)
            assert len(description) > 0


class TestCompareModels:
    """Tests for model comparison."""
    
    def test_find_best_model(self):
        """Test finding the best model by metric."""
        results = {
            'model_a': {'mape': 15.0, 'mae': 5.0},
            'model_b': {'mape': 10.0, 'mae': 6.0},
            'model_c': {'mape': 20.0, 'mae': 4.0}
        }
        
        best = compare_models(results, metric='mape')
        assert best == 'model_b'
        
        best_mae = compare_models(results, metric='mae')
        assert best_mae == 'model_c'
    
    def test_empty_results(self):
        """Test with empty results."""
        best = compare_models({}, metric='mape')
        assert best is None


class TestGenerateMetricsReport:
    """Tests for metrics report generation."""
    
    def test_report_structure(self, sample_predictions):
        """Test report structure."""
        y_true, y_pred = sample_predictions
        
        report = generate_metrics_report(y_true, y_pred, model_name="Test")
        
        assert 'model' in report
        assert 'metrics' in report
        assert 'quality' in report
        assert 'summary' in report
    
    def test_report_with_product_name(self, sample_predictions):
        """Test report includes product name."""
        y_true, y_pred = sample_predictions
        
        report = generate_metrics_report(
            y_true, y_pred, 
            model_name="Prophet",
            product_name="Poulet Frais"
        )
        
        assert report['product'] == "Poulet Frais"
    
    def test_report_summary_values(self, sample_predictions):
        """Test summary values in report."""
        y_true, y_pred = sample_predictions
        
        report = generate_metrics_report(y_true, y_pred)
        
        assert report['summary']['n_samples'] == len(y_true)
        assert report['summary']['n_nonzero'] <= len(y_true)


class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_single_value(self):
        """Test metrics with single value."""
        y_true = np.array([100])
        y_pred = np.array([90])
        
        metrics = calculate_metrics(y_true, y_pred)
        assert metrics['mae'] == 10
        assert metrics['mape'] == 10.0
    
    def test_identical_values(self):
        """Test metrics when all values are identical."""
        y_true = np.array([50, 50, 50, 50])
        y_pred = np.array([50, 50, 50, 50])
        
        metrics = calculate_metrics(y_true, y_pred)
        assert metrics['mae'] == 0
        assert metrics['mape'] == 0
        assert metrics['rmse'] == 0
    
    def test_list_input(self):
        """Test that list inputs work (converted to arrays)."""
        y_true = [10, 20, 30]
        y_pred = [11, 19, 32]
        
        mae = calculate_mae(y_true, y_pred)
        assert mae > 0
