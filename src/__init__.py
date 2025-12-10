"""
Hospital Stock Management AI - Mont Vert Clinic
================================================

Package for hospital stock prediction using Prophet time series forecasting.

Modules:
    - data_loader: Data loading and preprocessing
    - model: Prophet model training and prediction
    - metrics: Performance metrics calculation
    - visualization: Charts and plots generation
    - cli: Command line interface
"""

__version__ = "1.0.0"
__author__ = "Master EISI - Clinique du Mont Vert"

from .data_loader import load_dataset, prepare_prophet_data, train_test_split
from .model import train_prophet_model, predict, predict_future
from .metrics import calculate_metrics, calculate_mape, print_metrics
from .visualization import plot_predictions, plot_seasonality

__all__ = [
    "load_dataset",
    "prepare_prophet_data",
    "train_test_split",
    "train_prophet_model",
    "predict",
    "predict_future",
    "calculate_metrics",
    "calculate_mape",
    "print_metrics",
    "plot_predictions",
    "plot_seasonality",
]
