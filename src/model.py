"""
Prophet Model Training Module
=============================

Functions for training and using Prophet models for stock prediction.
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
from datetime import timedelta

# Prophet import with availability check
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("⚠️ Prophet not installed. Run: pip install prophet")


def check_prophet_available() -> bool:
    """Check if Prophet is available."""
    return PROPHET_AVAILABLE


def create_prophet_model(
    daily_seasonality: bool = False,
    weekly_seasonality: bool = True,
    yearly_seasonality: bool = True,
    seasonality_mode: str = 'additive',
    interval_width: float = 0.85,
    changepoint_prior_scale: float = 0.05,
    regressors: Optional[List[str]] = None
) -> "Prophet":
    """
    Create a configured Prophet model.
    
    Args:
        daily_seasonality: Enable daily seasonality
        weekly_seasonality: Enable weekly seasonality
        yearly_seasonality: Enable yearly seasonality
        seasonality_mode: 'additive' or 'multiplicative'
        interval_width: Width of uncertainty intervals
        changepoint_prior_scale: Flexibility of trend changes
        regressors: List of regressor names to add
        
    Returns:
        Configured Prophet model instance
    """
    if not PROPHET_AVAILABLE:
        raise ImportError("Prophet is not installed. Run: pip install prophet")
    
    model = Prophet(
        daily_seasonality=daily_seasonality,
        weekly_seasonality=weekly_seasonality,
        yearly_seasonality=yearly_seasonality,
        seasonality_mode=seasonality_mode,
        interval_width=interval_width,
        changepoint_prior_scale=changepoint_prior_scale
    )
    
    # Add regressors if specified
    if regressors:
        for reg in regressors:
            model.add_regressor(reg)
            print(f"   📊 Régresseur ajouté: {reg}")
    
    return model


def train_prophet_model(
    train_df: pd.DataFrame,
    daily_seasonality: bool = False,
    weekly_seasonality: bool = True,
    yearly_seasonality: bool = True,
    seasonality_mode: str = 'additive',
    interval_width: float = 0.85,
    changepoint_prior_scale: float = 0.05,
    regressors: Optional[List[str]] = None,
    verbose: bool = True
) -> "Prophet":
    """
    Train a Prophet model on the provided data.
    
    Args:
        train_df: Training DataFrame with 'ds', 'y' columns
        daily_seasonality: Enable daily seasonality
        weekly_seasonality: Enable weekly seasonality
        yearly_seasonality: Enable yearly seasonality
        seasonality_mode: 'additive' or 'multiplicative'
        interval_width: Width of uncertainty intervals
        changepoint_prior_scale: Flexibility of trend changes
        regressors: List of regressor column names
        verbose: Print progress messages
        
    Returns:
        Trained Prophet model
    """
    if verbose:
        print("🤖 Entraînement du modèle Prophet...")
        print("⏳ Cela peut prendre 1-2 minutes...\n")
    
    model = create_prophet_model(
        daily_seasonality=daily_seasonality,
        weekly_seasonality=weekly_seasonality,
        yearly_seasonality=yearly_seasonality,
        seasonality_mode=seasonality_mode,
        interval_width=interval_width,
        changepoint_prior_scale=changepoint_prior_scale,
        regressors=regressors
    )
    
    model.fit(train_df)
    
    if verbose:
        print("✅ Modèle entraîné !")
    
    return model


def predict(
    model: "Prophet",
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Make predictions using a trained Prophet model.
    
    Args:
        model: Trained Prophet model
        df: DataFrame with 'ds' column (and regressors if used)
        
    Returns:
        DataFrame with predictions
    """
    return model.predict(df)


def predict_future(
    model: "Prophet",
    periods: int = 30,
    freq: str = 'D',
    include_history: bool = False,
    future_regressors: Optional[pd.DataFrame] = None,
    start_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Generate future predictions.
    
    Args:
        model: Trained Prophet model
        periods: Number of periods to predict
        freq: Frequency of predictions ('D' for daily)
        include_history: Include historical data in output
        future_regressors: DataFrame with future regressor values
        start_date: Start date for predictions (default: today if after training data)
        
    Returns:
        DataFrame with future predictions
    """
    # If start_date is provided, create custom future dataframe
    if start_date:
        start = pd.to_datetime(start_date)
        future = pd.DataFrame({
            'ds': pd.date_range(start=start, periods=periods, freq=freq)
        })
    else:
        # Default: use today's date if it's after the training data
        from datetime import datetime
        today = pd.to_datetime(datetime.now().date())
        last_training_date = model.history['ds'].max()
        
        if today > last_training_date:
            # Start from today
            future = pd.DataFrame({
                'ds': pd.date_range(start=today, periods=periods, freq=freq)
            })
        else:
            # Use Prophet's default (continue from training data)
            future = model.make_future_dataframe(
                periods=periods,
                freq=freq,
                include_history=include_history
            )
    
    # Add regressors if provided
    if future_regressors is not None:
        for col in future_regressors.columns:
            if col != 'ds':
                future = future.merge(
                    future_regressors[['ds', col]], 
                    on='ds', 
                    how='left'
                )
                future[col] = future[col].fillna(method='ffill').fillna(0)
    
    predictions = model.predict(future)
    
    # Ensure non-negative predictions for stock quantities
    predictions['yhat'] = predictions['yhat'].clip(lower=0)
    predictions['yhat_lower'] = predictions['yhat_lower'].clip(lower=0)
    
    return predictions


def get_regressor_coefficients(model: "Prophet") -> Dict[str, Dict[str, float]]:
    """
    Extract regressor coefficients from a trained model.
    
    Args:
        model: Trained Prophet model with regressors
        
    Returns:
        Dictionary with regressor names and their coefficients
    """
    coefficients = {}
    
    if hasattr(model, 'extra_regressors') and model.extra_regressors:
        regressor_names = list(model.extra_regressors.keys())
        
        for name in regressor_names:
            if hasattr(model, 'params') and 'beta' in model.params:
                idx = regressor_names.index(name)
                if idx < len(model.params['beta'].flatten()):
                    coef = model.params['beta'].flatten()[idx]
                    coefficients[name] = {
                        'coefficient': float(coef),
                        'impact': 'positive' if coef > 0 else 'negative'
                    }
    
    return coefficients


def model_summary(model: "Prophet") -> Dict[str, Any]:
    """
    Get a summary of the trained model.
    
    Args:
        model: Trained Prophet model
        
    Returns:
        Dictionary with model information
    """
    summary = {
        "seasonality_mode": model.seasonality_mode,
        "changepoint_prior_scale": model.changepoint_prior_scale,
        "interval_width": model.interval_width,
        "daily_seasonality": model.daily_seasonality,
        "weekly_seasonality": model.weekly_seasonality,
        "yearly_seasonality": model.yearly_seasonality,
    }
    
    if hasattr(model, 'extra_regressors'):
        summary["regressors"] = list(model.extra_regressors.keys())
    
    if hasattr(model, 'changepoints'):
        summary["n_changepoints"] = len(model.changepoints)
    
    return summary
