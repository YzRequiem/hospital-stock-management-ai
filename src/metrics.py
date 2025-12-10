"""
Metrics Calculation Module
==========================

Functions for calculating prediction performance metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple


def calculate_mape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    exclude_zeros: bool = True
) -> float:
    """
    Calculate Mean Absolute Percentage Error (MAPE).
    
    Args:
        y_true: Array of actual values
        y_pred: Array of predicted values
        exclude_zeros: Exclude zero values to avoid division by zero
        
    Returns:
        MAPE as a percentage
        
    Note:
        When exclude_zeros=True, only days with actual consumption > 0
        are included in the calculation. This provides a more meaningful
        metric for stock prediction where many days may have zero consumption.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if exclude_zeros:
        mask = y_true > 0
        if mask.sum() == 0:
            return 0.0
        y_true = y_true[mask]
        y_pred = y_pred[mask]
    else:
        # Add small epsilon to avoid division by zero
        y_true = y_true + 1e-10
    
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return float(mape)


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error (MAE).
    
    Args:
        y_true: Array of actual values
        y_pred: Array of predicted values
        
    Returns:
        MAE value
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Root Mean Square Error (RMSE).
    
    Args:
        y_true: Array of actual values
        y_pred: Array of predicted values
        
    Returns:
        RMSE value
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate R² (coefficient of determination).
    
    Args:
        y_true: Array of actual values
        y_pred: Array of predicted values
        
    Returns:
        R² score between 0 and 1
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    if ss_tot == 0:
        return 0.0
    
    return float(1 - (ss_res / ss_tot))


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    exclude_zeros_for_mape: bool = True
) -> Dict[str, float]:
    """
    Calculate all prediction metrics.
    
    Args:
        y_true: Array of actual values
        y_pred: Array of predicted values
        exclude_zeros_for_mape: Exclude zeros when calculating MAPE
        
    Returns:
        Dictionary with MAE, MAPE, RMSE, and R² scores
    """
    return {
        "mae": calculate_mae(y_true, y_pred),
        "mape": calculate_mape(y_true, y_pred, exclude_zeros=exclude_zeros_for_mape),
        "rmse": calculate_rmse(y_true, y_pred),
        "r2": calculate_r2(y_true, y_pred)
    }


def interpret_mape(mape: float) -> Tuple[str, str]:
    """
    Interpret MAPE value with quality assessment.
    
    Args:
        mape: MAPE value as percentage
        
    Returns:
        Tuple of (quality_level, description)
    """
    if mape < 10:
        return ("excellent", "✅ Excellente précision (MAPE < 10%)")
    elif mape < 15:
        return ("very_good", "✅ Très bonne précision (MAPE < 15%)")
    elif mape < 25:
        return ("good", "✅ Bonne précision (MAPE < 25%)")
    elif mape < 50:
        return ("acceptable", "⚠️ Précision acceptable (MAPE < 50%)")
    else:
        return ("poor", "❌ Précision insuffisante (MAPE > 50%)")


def print_metrics(
    metrics: Dict[str, float],
    unit: str = "kg",
    title: str = "MÉTRIQUES DE PERFORMANCE"
) -> None:
    """
    Print formatted metrics report.
    
    Args:
        metrics: Dictionary with metric values
        unit: Unit for MAE/RMSE display
        title: Report title
    """
    print(f"\n📊 {title}")
    print("=" * 70)
    print(f"MAE  (Erreur Absolue Moyenne)    : {metrics['mae']:.2f} {unit}")
    print(f"MAPE (Erreur Relative Moyenne)   : {metrics['mape']:.2f}%")
    print(f"RMSE (Erreur Quadratique Moyenne): {metrics['rmse']:.2f} {unit}")
    print(f"R²   (Coefficient Détermination) : {metrics['r2']:.4f}")
    print("=" * 70)
    
    quality, description = interpret_mape(metrics['mape'])
    print(f"\n💡 Interprétation: {description}")
    print(f"   Le modèle se trompe en moyenne de ±{metrics['mae']:.2f} {unit}")


def compare_models(
    results: Dict[str, Dict[str, float]],
    metric: str = "mape"
) -> str:
    """
    Compare multiple models and return the best one.
    
    Args:
        results: Dictionary mapping model names to their metrics
        metric: Metric to use for comparison (lower is better)
        
    Returns:
        Name of the best model
    """
    best_model = None
    best_value = float('inf')
    
    for model_name, metrics in results.items():
        if metric in metrics and metrics[metric] < best_value:
            best_value = metrics[metric]
            best_model = model_name
    
    return best_model


def generate_metrics_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Prophet",
    product_name: Optional[str] = None
) -> Dict:
    """
    Generate a complete metrics report.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        model_name: Name of the model
        product_name: Optional product name
        
    Returns:
        Dictionary with complete report data
    """
    metrics = calculate_metrics(y_true, y_pred)
    quality, description = interpret_mape(metrics['mape'])
    
    report = {
        "model": model_name,
        "product": product_name,
        "metrics": metrics,
        "quality": {
            "level": quality,
            "description": description
        },
        "summary": {
            "n_samples": len(y_true),
            "n_nonzero": int((np.asarray(y_true) > 0).sum()),
            "mean_actual": float(np.mean(y_true)),
            "mean_predicted": float(np.mean(y_pred))
        }
    }
    
    return report
