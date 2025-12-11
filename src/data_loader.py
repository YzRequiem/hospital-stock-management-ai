"""
Data Loading and Preprocessing Module
======================================

Functions for loading hospital stock datasets and preparing data for Prophet.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta
from typing import Optional, Tuple, List


def load_dataset(
    filepath: str,
    encoding: str = "utf-8",
    separator: str = ","
) -> pd.DataFrame:
    """
    Load a hospital stock dataset from CSV file.
    
    Args:
        filepath: Path to the CSV file
        encoding: File encoding (default: utf-8)
        separator: CSV separator (default: comma)
        
    Returns:
        DataFrame with the loaded data
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file format is invalid
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {filepath}")
    
    # Try different encodings if default fails
    encodings_to_try = [encoding, "utf-8", "latin-1", "cp1252"]
    
    for enc in encodings_to_try:
        try:
            df = pd.read_csv(filepath, encoding=enc, sep=separator)
            print(f"✅ Dataset chargé: {len(df)} lignes, {len(df.columns)} colonnes")
            return df
        except UnicodeDecodeError:
            continue
    
    raise ValueError(f"Unable to read file with any encoding: {filepath}")


def prepare_prophet_data(
    df: pd.DataFrame,
    date_column: str = "date",
    target_column: str = "quantite_consommee",
    product_filter: Optional[str] = None,
    product_column: str = "produit",
    regressors: Optional[List[str]] = None,
    fill_missing_dates: bool = True
) -> pd.DataFrame:
    """
    Prepare data for Prophet model training.
    
    Args:
        df: Input DataFrame
        date_column: Name of the date column
        target_column: Name of the target variable column
        product_filter: Optional product name to filter
        product_column: Name of the product column
        regressors: Optional list of regressor column names
        fill_missing_dates: Whether to fill missing dates with 0
        
    Returns:
        DataFrame with 'ds', 'y' columns (+ regressors if specified)
    """
    df = df.copy()
    
    # Filter by product if specified
    if product_filter and product_column in df.columns:
        df = df[df[product_column] == product_filter]
        print(f"📦 Produit filtré: {product_filter} ({len(df)} lignes)")
    
    # Parse dates
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    df = df.dropna(subset=[date_column])
    
    # Aggregate by date
    agg_dict = {target_column: 'sum'}
    if regressors:
        for reg in regressors:
            if reg in df.columns:
                agg_dict[reg] = 'mean'
    
    daily = df.groupby(date_column).agg(agg_dict).reset_index()
    
    # Rename to Prophet format
    prophet_df = daily.rename(columns={
        date_column: 'ds',
        target_column: 'y'
    })
    
    # Fill missing dates
    if fill_missing_dates and len(prophet_df) > 0:
        date_range = pd.date_range(
            start=prophet_df['ds'].min(),
            end=prophet_df['ds'].max(),
            freq='D'
        )
        full_dates = pd.DataFrame({'ds': date_range})
        prophet_df = full_dates.merge(prophet_df, on='ds', how='left')
        prophet_df['y'] = prophet_df['y'].fillna(0)
        
        # Fill regressors with forward/backward fill
        if regressors:
            for reg in regressors:
                if reg in prophet_df.columns:
                    prophet_df[reg] = prophet_df[reg].fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    print(f"✅ {len(prophet_df)} jours préparés pour Prophet")
    return prophet_df


def train_test_split(
    df: pd.DataFrame,
    test_ratio: float = 0.2,
    date_column: str = 'ds'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into training and test sets (chronological).
    
    Args:
        df: Input DataFrame with 'ds' column
        test_ratio: Proportion of data for testing (default: 0.2)
        date_column: Name of the date column
        
    Returns:
        Tuple of (train_df, test_df)
    """
    df = df.sort_values(date_column)
    split_date = df[date_column].max() - timedelta(days=int(len(df) * test_ratio))
    
    train = df[df[date_column] <= split_date].copy()
    test = df[df[date_column] > split_date].copy()
    
    print(f"Train: {len(train)} jours ({train[date_column].min().date()} → {train[date_column].max().date()})")
    print(f"Test:  {len(test)} jours ({test[date_column].min().date()} → {test[date_column].max().date()})")
    
    return train, test


def get_dataset_info(df: pd.DataFrame) -> dict:
    """
    Get summary information about a dataset.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with dataset statistics
    """
    info = {
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": list(df.columns),
        "dtypes": {k: str(v) for k, v in df.dtypes.to_dict().items()},
        "missing_values": df.isnull().sum().to_dict(),
        "memory_mb": df.memory_usage(deep=True).sum() / 1024 / 1024
    }
    
    # Date range - check by column name first, then by dtype
    date_col = None
    for col_name in ['date', 'ds', 'Date', 'DATE']:
        if col_name in df.columns:
            date_col = col_name
            break
    
    # Fallback to datetime columns
    if date_col is None:
        date_cols = df.select_dtypes(include=['datetime64']).columns
        if len(date_cols) > 0:
            date_col = date_cols[0]
    
    if date_col:
        try:
            dates = pd.to_datetime(df[date_col], errors='coerce')
            dates = dates.dropna()
            if len(dates) > 0:
                info["date_range"] = {
                    "start": dates.min(),
                    "end": dates.max(),
                    "days": (dates.max() - dates.min()).days
                }
        except Exception:
            pass
    
    return info
