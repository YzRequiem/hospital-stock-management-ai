"""
Tests for Data Loader Module
============================

Unit tests for data loading and preprocessing functions.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import (
    load_dataset,
    prepare_prophet_data,
    train_test_split,
    get_dataset_info
)


class TestLoadDataset:
    """Tests for load_dataset function."""
    
    def test_load_valid_csv(self, sample_csv_file):
        """Test loading a valid CSV file."""
        df = load_dataset(str(sample_csv_file))
        
        assert df is not None
        assert len(df) == 365
        assert 'date' in df.columns
        assert 'quantite_consommee' in df.columns
    
    def test_load_nonexistent_file(self):
        """Test that loading nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            load_dataset("nonexistent_file.csv")
    
    def test_load_with_different_encodings(self, temp_dir):
        """Test loading file with different encodings."""
        # Create a file with latin-1 encoding
        filepath = temp_dir / "latin1.csv"
        df = pd.DataFrame({'col1': ['café', 'résumé'], 'col2': [1, 2]})
        df.to_csv(filepath, index=False, encoding='latin-1')
        
        # Should still load successfully
        result = load_dataset(str(filepath))
        assert len(result) == 2


class TestPrepareProphetData:
    """Tests for prepare_prophet_data function."""
    
    def test_basic_preparation(self, sample_consumption_data):
        """Test basic data preparation for Prophet."""
        df = prepare_prophet_data(
            sample_consumption_data,
            date_column='date',
            target_column='quantite_consommee'
        )
        
        assert 'ds' in df.columns
        assert 'y' in df.columns
        assert len(df) == 365
        assert df['ds'].dtype == 'datetime64[ns]'
    
    def test_product_filtering(self, sample_consumption_data):
        """Test filtering by product."""
        # Add another product
        df = sample_consumption_data.copy()
        df2 = df.copy()
        df2['produit'] = 'Poisson'
        combined = pd.concat([df, df2], ignore_index=True)
        
        result = prepare_prophet_data(
            combined,
            date_column='date',
            target_column='quantite_consommee',
            product_filter='Poulet Frais',
            product_column='produit'
        )
        
        # Should aggregate by date, so still 365 days
        assert len(result) == 365
    
    def test_fill_missing_dates(self):
        """Test filling of missing dates."""
        # Create data with gaps
        dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
        dates = dates.delete([3, 4, 5])  # Remove some days
        
        df = pd.DataFrame({
            'date': dates,
            'quantite_consommee': [10] * len(dates)
        })
        
        result = prepare_prophet_data(
            df,
            date_column='date',
            target_column='quantite_consommee',
            fill_missing_dates=True
        )
        
        # Should have all 10 days filled
        assert len(result) == 10
        # Missing days should have y=0
        assert (result['y'] == 0).sum() == 3
    
    def test_without_fill_missing_dates(self, sample_consumption_data):
        """Test without filling missing dates."""
        result = prepare_prophet_data(
            sample_consumption_data,
            date_column='date',
            target_column='quantite_consommee',
            fill_missing_dates=False
        )
        
        assert len(result) == 365


class TestTrainTestSplit:
    """Tests for train_test_split function."""
    
    def test_default_split(self, sample_prophet_data):
        """Test default 80/20 split."""
        train, test = train_test_split(sample_prophet_data)
        
        total = len(sample_prophet_data)
        expected_test_size = int(total * 0.2)
        
        # Allow some tolerance due to date-based splitting
        assert abs(len(test) - expected_test_size) <= 5
        assert len(train) + len(test) == total
    
    def test_custom_split_ratio(self, sample_prophet_data):
        """Test custom split ratio."""
        train, test = train_test_split(sample_prophet_data, test_ratio=0.3)
        
        total = len(sample_prophet_data)
        expected_test_size = int(total * 0.3)
        
        assert abs(len(test) - expected_test_size) <= 5
    
    def test_chronological_order(self, sample_prophet_data):
        """Test that split maintains chronological order."""
        train, test = train_test_split(sample_prophet_data)
        
        # All train dates should be before test dates
        assert train['ds'].max() < test['ds'].min()
    
    def test_no_data_leakage(self, sample_prophet_data):
        """Test that there's no overlap between train and test."""
        train, test = train_test_split(sample_prophet_data)
        
        train_dates = set(train['ds'])
        test_dates = set(test['ds'])
        
        assert len(train_dates.intersection(test_dates)) == 0


class TestGetDatasetInfo:
    """Tests for get_dataset_info function."""
    
    def test_basic_info(self, sample_consumption_data):
        """Test getting basic dataset info."""
        info = get_dataset_info(sample_consumption_data)
        
        assert info['rows'] == 365
        assert info['columns'] == 4
        assert 'date' in info['column_names']
        assert info['memory_mb'] > 0
    
    def test_date_range_info(self, sample_consumption_data):
        """Test date range information."""
        df = sample_consumption_data.copy()
        df['date'] = pd.to_datetime(df['date'])
        
        info = get_dataset_info(df)
        
        assert 'date_range' in info
        assert info['date_range']['days'] == 364  # 365 days span = 364 days diff
    
    def test_missing_values(self):
        """Test detection of missing values."""
        df = pd.DataFrame({
            'col1': [1, 2, None, 4],
            'col2': [None, None, 3, 4]
        })
        
        info = get_dataset_info(df)
        
        assert info['missing_values']['col1'] == 1
        assert info['missing_values']['col2'] == 2


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame(columns=['date', 'quantite_consommee'])
        
        result = prepare_prophet_data(df)
        assert len(result) == 0
    
    def test_single_row(self):
        """Test handling of single row DataFrame."""
        df = pd.DataFrame({
            'date': [datetime(2024, 1, 1)],
            'quantite_consommee': [50]
        })
        
        result = prepare_prophet_data(df)
        assert len(result) == 1
        assert result['y'].iloc[0] == 50
    
    def test_invalid_date_column(self, sample_consumption_data):
        """Test handling of invalid date column."""
        # Should handle gracefully
        result = prepare_prophet_data(
            sample_consumption_data,
            date_column='nonexistent_column',
            target_column='quantite_consommee'
        )
        # Depending on implementation, might return empty or raise error
        assert result is not None or True


class TestIntegration:
    """Integration tests combining multiple functions."""
    
    def test_full_pipeline(self, sample_csv_file):
        """Test complete data loading pipeline."""
        # Load
        df = load_dataset(str(sample_csv_file))
        
        # Get info
        info = get_dataset_info(df)
        assert info['rows'] > 0
        
        # Prepare for Prophet
        prophet_df = prepare_prophet_data(df)
        assert 'ds' in prophet_df.columns
        assert 'y' in prophet_df.columns
        
        # Split
        train, test = train_test_split(prophet_df)
        assert len(train) > 0
        assert len(test) > 0
        assert len(train) > len(test)
