"""
Pytest Configuration and Fixtures
==================================

Shared fixtures for testing the hospital stock management system.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import tempfile
import shutil


# =============================================================================
# Path Fixtures
# =============================================================================

@pytest.fixture
def project_root():
    """Get the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def data_dir(project_root):
    """Get the data directory."""
    return project_root / "data"


@pytest.fixture
def config_dir(project_root):
    """Get the config directory."""
    return project_root / "config"


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    # Cleanup after test
    shutil.rmtree(temp_path, ignore_errors=True)


# =============================================================================
# Sample Data Fixtures
# =============================================================================

@pytest.fixture
def sample_consumption_data():
    """Create sample consumption data for testing."""
    np.random.seed(42)
    
    # Generate 365 days of data
    dates = pd.date_range(start='2024-01-01', periods=365, freq='D')
    
    # Base consumption with weekly pattern
    base = 50
    weekly_pattern = [1.2, 1.1, 1.0, 1.0, 0.9, 0.5, 0.3]  # Mon-Sun
    
    data = []
    for date in dates:
        day_of_week = date.dayofweek
        consumption = base * weekly_pattern[day_of_week]
        # Add some noise
        consumption += np.random.normal(0, 5)
        consumption = max(0, consumption)
        
        data.append({
            'date': date,
            'produit': 'Poulet Frais',
            'quantite_consommee': consumption,
            'categorie': 'viande'
        })
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_prophet_data(sample_consumption_data):
    """Create sample data in Prophet format (ds, y)."""
    df = sample_consumption_data.copy()
    return pd.DataFrame({
        'ds': df['date'],
        'y': df['quantite_consommee']
    })


@pytest.fixture
def sample_enriched_data():
    """Create sample enriched data with regressors."""
    np.random.seed(42)
    
    dates = pd.date_range(start='2024-01-01', periods=365, freq='D')
    
    data = []
    for i, date in enumerate(dates):
        # Simulate seasonal temperature
        day_of_year = date.dayofyear
        temperature = 15 + 10 * np.sin(2 * np.pi * (day_of_year - 100) / 365)
        temperature += np.random.normal(0, 3)
        
        # Occupancy rate
        occupancy = 0.75 + np.random.uniform(-0.15, 0.15)
        
        # Number of patients
        patients = int(100 * occupancy + np.random.normal(0, 10))
        
        # Epidemic (winter months)
        epidemic = 1 if (date.month in [12, 1, 2] and np.random.random() < 0.3) else 0
        
        # Holiday
        holiday = 1 if date.dayofweek >= 5 else 0
        
        # Consumption based on factors
        base = 50
        consumption = base * (0.8 + 0.4 * occupancy)
        consumption += temperature * 0.5
        consumption -= epidemic * 10
        consumption -= holiday * 15
        consumption += np.random.normal(0, 5)
        consumption = max(0, consumption)
        
        data.append({
            'date': date,
            'produit': 'Poulet Frais',
            'quantite_consommee': consumption,
            'temperature': temperature,
            'taux_occupation': occupancy,
            'nb_patients_jour': patients,
            'epidemie_active': epidemic,
            'jour_ferie': holiday,
            'periode_covid': 0
        })
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_predictions():
    """Create sample prediction values."""
    np.random.seed(42)
    n = 100
    
    y_true = np.random.uniform(20, 80, n)
    # Add some zeros
    y_true[::10] = 0
    
    # Predictions with some error
    y_pred = y_true + np.random.normal(0, 10, n)
    y_pred = np.clip(y_pred, 0, None)
    
    return y_true, y_pred


@pytest.fixture
def sample_csv_file(temp_dir, sample_consumption_data):
    """Create a temporary CSV file with sample data."""
    filepath = temp_dir / "test_data.csv"
    sample_consumption_data.to_csv(filepath, index=False)
    return filepath


# =============================================================================
# Configuration Fixtures
# =============================================================================

@pytest.fixture
def sample_prophet_config():
    """Sample Prophet configuration dictionary."""
    return {
        'daily_seasonality': False,
        'weekly_seasonality': True,
        'yearly_seasonality': True,
        'seasonality_mode': 'additive',
        'interval_width': 0.85,
        'changepoint_prior_scale': 0.05
    }


@pytest.fixture
def sample_product_config():
    """Sample product configuration."""
    return {
        'name': 'Poulet Frais',
        'category': 'viande',
        'dlc_days': 2,
        'priority': 'high',
        'unit': 'kg',
        'min_stock': 10,
        'max_stock': 100,
        'reorder_point': 20
    }


# =============================================================================
# Mock Fixtures
# =============================================================================

@pytest.fixture
def mock_prophet_model(sample_prophet_data):
    """Create a mock trained Prophet model (if Prophet is available)."""
    try:
        from prophet import Prophet
        
        model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True
        )
        
        # Use only first 300 days for faster training
        train_data = sample_prophet_data.head(300)
        model.fit(train_data)
        
        return model
    except ImportError:
        pytest.skip("Prophet not installed")


# =============================================================================
# Helpers
# =============================================================================

def assert_dataframe_equal(df1, df2, check_dtype=False):
    """Helper to compare DataFrames in tests."""
    pd.testing.assert_frame_equal(df1, df2, check_dtype=check_dtype)


def assert_close(a, b, rel_tol=1e-5, abs_tol=1e-8):
    """Helper to compare floating point numbers."""
    assert abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)
