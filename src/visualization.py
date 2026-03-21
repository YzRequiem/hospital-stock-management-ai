"""
Visualization Module
====================

Functions for creating charts and plots for stock prediction analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Optional, Tuple, List
from pathlib import Path


# Set default style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 10


def plot_predictions(
    actual: pd.DataFrame,
    predictions: pd.DataFrame,
    title: str = "Prédictions vs Réalité",
    save_path: Optional[str] = None,
    show_confidence: bool = True,
    figsize: Tuple[int, int] = (14, 6)
) -> plt.Figure:
    """
    Plot actual vs predicted values with confidence intervals.
    
    Args:
        actual: DataFrame with 'ds' and 'y' columns
        predictions: Prophet predictions DataFrame
        title: Plot title
        save_path: Optional path to save the figure
        show_confidence: Show confidence intervals
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot actual values
    ax.plot(
        actual['ds'], 
        actual['y'], 
        'b-', 
        label='Réel', 
        alpha=0.7,
        linewidth=1.5
    )
    
    # Plot predictions
    ax.plot(
        predictions['ds'], 
        predictions['yhat'], 
        'r--', 
        label='Prédit', 
        alpha=0.8,
        linewidth=1.5
    )
    
    # Confidence interval
    if show_confidence and 'yhat_lower' in predictions.columns:
        ax.fill_between(
            predictions['ds'],
            predictions['yhat_lower'],
            predictions['yhat_upper'],
            color='red',
            alpha=0.2,
            label='Intervalle de confiance'
        )
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Quantité (kg)')
    ax.set_title(title)
    ax.legend(loc='upper left')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Graphique sauvegardé: {save_path}")
    
    return fig


def plot_seasonality(
    model,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 10)
) -> plt.Figure:
    """
    Plot Prophet model seasonality components.
    
    Args:
        model: Trained Prophet model
        save_path: Optional path to save the figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    from prophet.plot import plot_components
    
    # Create future dataframe for component plotting
    future = model.make_future_dataframe(periods=0)
    forecast = model.predict(future)
    
    fig = model.plot_components(forecast)
    fig.set_size_inches(figsize)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Saisonnalité sauvegardée: {save_path}")
    
    return fig


def plot_consumption_distribution(
    data: pd.Series,
    title: str = "Distribution de la Consommation",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5)
) -> plt.Figure:
    """
    Plot histogram and boxplot of consumption data.
    
    Args:
        data: Series of consumption values
        title: Plot title
        save_path: Optional path to save the figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Histogram
    axes[0].hist(data, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0].axvline(data.mean(), color='red', linestyle='--', label=f'Moyenne: {data.mean():.2f}')
    axes[0].axvline(data.median(), color='orange', linestyle='--', label=f'Médiane: {data.median():.2f}')
    axes[0].set_xlabel('Quantité (kg)')
    axes[0].set_ylabel('Fréquence')
    axes[0].set_title('Distribution')
    axes[0].legend()
    
    # Boxplot
    axes[1].boxplot(data, vert=True)
    axes[1].set_ylabel('Quantité (kg)')
    axes[1].set_title('Boxplot')
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Distribution sauvegardée: {save_path}")
    
    return fig


def plot_weekly_pattern(
    df: pd.DataFrame,
    date_col: str = 'ds',
    value_col: str = 'y',
    title: str = "Pattern Hebdomadaire",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 5)
) -> plt.Figure:
    """
    Plot average consumption by day of week.
    
    Args:
        df: DataFrame with date and value columns
        date_col: Name of date column
        value_col: Name of value column
        title: Plot title
        save_path: Optional path to save the figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    df = df.copy()
    df['day_of_week'] = pd.to_datetime(df[date_col]).dt.dayofweek
    
    days = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']
    daily_avg = df.groupby('day_of_week')[value_col].mean()
    
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(range(7), daily_avg.values, color='steelblue', edgecolor='black')
    
    # Highlight weekend
    bars[5].set_color('lightcoral')
    bars[6].set_color('lightcoral')
    
    ax.set_xticks(range(7))
    ax.set_xticklabels(days, rotation=45)
    ax.set_xlabel('Jour de la semaine')
    ax.set_ylabel('Consommation moyenne (kg)')
    ax.set_title(title)
    
    # Add value labels
    for i, v in enumerate(daily_avg.values):
        ax.text(i, v + 0.5, f'{v:.1f}', ha='center', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Pattern hebdomadaire sauvegardé: {save_path}")
    
    return fig


def plot_monthly_trend(
    df: pd.DataFrame,
    date_col: str = 'ds',
    value_col: str = 'y',
    title: str = "Tendance Mensuelle",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5)
) -> plt.Figure:
    """
    Plot monthly consumption trend.
    
    Args:
        df: DataFrame with date and value columns
        date_col: Name of date column
        value_col: Name of value column
        title: Plot title
        save_path: Optional path to save the figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df['year_month'] = df[date_col].dt.to_period('M')
    
    monthly = df.groupby('year_month')[value_col].sum()
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(range(len(monthly)), monthly.values, 'b-o', linewidth=2, markersize=4)
    
    # Set x-axis labels
    tick_positions = list(range(0, len(monthly), max(1, len(monthly) // 12)))
    tick_labels = [str(monthly.index[i]) for i in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45)
    
    ax.set_xlabel('Mois')
    ax.set_ylabel('Consommation totale (kg)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Tendance mensuelle sauvegardée: {save_path}")
    
    return fig


def plot_regressor_impact(
    coefficients: dict,
    title: str = "Impact des Régresseurs",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot regressor coefficients as horizontal bar chart.
    
    Args:
        coefficients: Dictionary with regressor names and coefficient values
        title: Plot title
        save_path: Optional path to save the figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    names = list(coefficients.keys())
    values = [coefficients[n]['coefficient'] if isinstance(coefficients[n], dict) 
              else coefficients[n] for n in names]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = ['green' if v > 0 else 'red' for v in values]
    bars = ax.barh(names, values, color=colors, edgecolor='black', alpha=0.7)
    
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Coefficient')
    ax.set_title(title)
    
    # Add value labels
    for bar, val in zip(bars, values):
        x_pos = val + (0.01 if val >= 0 else -0.01)
        ax.text(x_pos, bar.get_y() + bar.get_height()/2, f'{val:.3f}', 
                va='center', ha='left' if val >= 0 else 'right', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Impact régresseurs sauvegardé: {save_path}")
    
    return fig


def plot_error_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dates: Optional[pd.Series] = None,
    title: str = "Analyse des Erreurs",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 8)
) -> plt.Figure:
    """
    Plot error analysis with residuals and error distribution.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        dates: Optional date series
        title: Plot title
        save_path: Optional path to save the figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    errors = y_true - y_pred
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Residuals over time/index
    x_axis = dates if dates is not None else range(len(errors))
    axes[0, 0].scatter(x_axis, errors, alpha=0.5, s=10)
    axes[0, 0].axhline(0, color='red', linestyle='--')
    axes[0, 0].set_xlabel('Date' if dates is not None else 'Index')
    axes[0, 0].set_ylabel('Erreur (kg)')
    axes[0, 0].set_title('Résidus')
    
    # Error histogram
    axes[0, 1].hist(errors, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(0, color='red', linestyle='--')
    axes[0, 1].set_xlabel('Erreur (kg)')
    axes[0, 1].set_ylabel('Fréquence')
    axes[0, 1].set_title('Distribution des erreurs')
    
    # Predicted vs Actual
    axes[1, 0].scatter(y_true, y_pred, alpha=0.5, s=10)
    max_val = max(y_true.max(), y_pred.max())
    axes[1, 0].plot([0, max_val], [0, max_val], 'r--', label='Parfait')
    axes[1, 0].set_xlabel('Valeur réelle (kg)')
    axes[1, 0].set_ylabel('Valeur prédite (kg)')
    axes[1, 0].set_title('Prédit vs Réel')
    axes[1, 0].legend()
    
    # Absolute percentage error distribution
    mask = y_true > 0
    if mask.sum() > 0:
        ape = np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]) * 100
        axes[1, 1].hist(ape[ape < 200], bins=50, edgecolor='black', alpha=0.7)  # Cap at 200% for visibility
        axes[1, 1].axvline(np.median(ape), color='red', linestyle='--', label=f'Médiane: {np.median(ape):.1f}%')
        axes[1, 1].set_xlabel('Erreur relative (%)')
        axes[1, 1].set_ylabel('Fréquence')
        axes[1, 1].set_title('Distribution APE (< 200%)')
        axes[1, 1].legend()
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Analyse erreurs sauvegardée: {save_path}")
    
    return fig


def close_all_figures():
    """Close all open matplotlib figures."""
    plt.close('all')
