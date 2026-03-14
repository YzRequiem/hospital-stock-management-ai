"""
Tests for Configuration Module
==============================

Unit tests for configuration loading and management.
"""

import pytest
import yaml
from pathlib import Path
import tempfile

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import (
    Settings,
    ProphetConfig,
    DataConfig,
    OutputConfig,
    load_yaml_config,
    get_dataset_path,
    get_results_path,
    create_results_directory,
    PROJECT_ROOT,
    CONFIG_DIR,
    DATA_DIR,
    RESULTS_DIR,
    DATASETS,
    REGRESSORS
)


class TestPaths:
    """Tests for path constants."""
    
    def test_project_root_exists(self):
        """Project root should exist."""
        assert PROJECT_ROOT.exists()
    
    def test_config_dir_exists(self):
        """Config directory should exist."""
        assert CONFIG_DIR.exists()
    
    def test_data_dir_exists(self):
        """Data directory should exist."""
        assert DATA_DIR.exists()
    
    def test_paths_are_absolute(self):
        """All paths should be absolute."""
        assert PROJECT_ROOT.is_absolute()
        assert CONFIG_DIR.is_absolute()
        assert DATA_DIR.is_absolute()


class TestProphetConfig:
    """Tests for ProphetConfig dataclass."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = ProphetConfig()
        
        assert config.daily_seasonality == False
        assert config.weekly_seasonality == True
        assert config.yearly_seasonality == True
        assert config.seasonality_mode == "additive"
        assert config.interval_width == 0.85
        assert config.changepoint_prior_scale == 0.05
    
    def test_custom_values(self):
        """Test custom configuration values."""
        config = ProphetConfig(
            daily_seasonality=True,
            interval_width=0.95
        )
        
        assert config.daily_seasonality == True
        assert config.interval_width == 0.95


class TestDataConfig:
    """Tests for DataConfig dataclass."""
    
    def test_default_values(self):
        """Test default data configuration."""
        config = DataConfig()
        
        assert config.date_column == "date"
        assert config.target_column == "quantite_consommee"
        assert config.product_column == "produit"
        assert config.test_ratio == 0.2
    
    def test_custom_values(self):
        """Test custom data configuration."""
        config = DataConfig(
            date_column="timestamp",
            test_ratio=0.3
        )
        
        assert config.date_column == "timestamp"
        assert config.test_ratio == 0.3


class TestOutputConfig:
    """Tests for OutputConfig dataclass."""
    
    def test_default_values(self):
        """Test default output configuration."""
        config = OutputConfig()
        
        assert config.save_plots == True
        assert config.plot_format == "png"
        assert config.plot_dpi == 150
        assert config.generate_report == True


class TestSettings:
    """Tests for main Settings class."""
    
    def test_default_settings(self):
        """Test default settings initialization."""
        settings = Settings()
        
        assert isinstance(settings.prophet, ProphetConfig)
        assert isinstance(settings.data, DataConfig)
        assert isinstance(settings.output, OutputConfig)
    
    def test_to_dict(self):
        """Test settings conversion to dictionary."""
        settings = Settings()
        d = settings.to_dict()
        
        assert 'prophet' in d
        assert 'data' in d
        assert 'output' in d
        
        assert d['prophet']['weekly_seasonality'] == True
        assert d['data']['test_ratio'] == 0.2
    
    def test_from_yaml(self, temp_dir):
        """Test loading settings from YAML file."""
        yaml_content = {
            'prophet': {
                'daily_seasonality': True,
                'interval_width': 0.90
            },
            'data': {
                'test_ratio': 0.25
            }
        }
        
        yaml_path = temp_dir / "test_settings.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f)
        
        settings = Settings.from_yaml(str(yaml_path))
        
        assert settings.prophet.daily_seasonality == True
        assert settings.prophet.interval_width == 0.90
        assert settings.data.test_ratio == 0.25


class TestLoadYamlConfig:
    """Tests for YAML configuration loading."""
    
    def test_load_products_config(self, config_dir):
        """Test loading products.yaml if it exists."""
        products_path = config_dir / "products.yaml"
        if products_path.exists():
            config = load_yaml_config("products")
            assert 'products' in config
    
    def test_load_model_params_config(self, config_dir):
        """Test loading model_params.yaml if it exists."""
        params_path = config_dir / "model_params.yaml"
        if params_path.exists():
            config = load_yaml_config("model_params")
            assert 'default' in config or 'prophet' in config or len(config) > 0
    
    def test_load_nonexistent_file(self):
        """Test loading nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            load_yaml_config("nonexistent_config")
    
    def test_auto_add_yaml_extension(self, config_dir):
        """Test that .yaml extension is auto-added."""
        # Should work without .yaml extension
        if (config_dir / "products.yaml").exists():
            config1 = load_yaml_config("products")
            config2 = load_yaml_config("products.yaml")
            assert config1 == config2


class TestGetDatasetPath:
    """Tests for get_dataset_path function."""
    
    def test_with_csv_extension(self):
        """Test path with .csv extension."""
        path = get_dataset_path("test.csv")
        assert path.suffix == ".csv"
        assert path.name == "test.csv"
    
    def test_without_csv_extension(self):
        """Test path without .csv extension (auto-added)."""
        path = get_dataset_path("test")
        assert path.suffix == ".csv"
        assert path.name == "test.csv"
    
    def test_path_in_data_dir(self):
        """Test that path is in data directory."""
        path = get_dataset_path("test")
        assert path.parent == DATA_DIR


class TestGetResultsPath:
    """Tests for get_results_path function."""
    
    def test_base_results_path(self):
        """Test getting base results path."""
        path = get_results_path()
        assert path == RESULTS_DIR
    
    def test_with_subdir(self, temp_dir):
        """Test creating subdirectory."""
        # This will create in actual results dir, so just test the logic
        subdir = "test_subdir_12345"
        path = get_results_path(subdir)
        
        assert subdir in str(path)
        # Clean up
        if path.exists():
            path.rmdir()


class TestCreateResultsDirectory:
    """Tests for create_results_directory function."""
    
    def test_creates_directory(self):
        """Test that directory is created."""
        path = create_results_directory("test_product")
        
        assert path.exists()
        assert path.is_dir()
        assert "test_product" in path.name
        
        # Clean up
        path.rmdir()
    
    def test_timestamp_in_name(self):
        """Test that timestamp is included in directory name."""
        path = create_results_directory("poulet_frais")
        
        # Should have format: analyse-poulet_frais-YYYYMMDD_HHMMSS
        assert "analyse-" in path.name
        assert "poulet_frais" in path.name
        
        # Clean up
        path.rmdir()
    
    def test_special_characters_handled(self):
        """Test that special characters are handled."""
        path = create_results_directory("Poulet Frais Spécial")
        
        # Should be cleaned up
        assert " " not in path.name
        
        # Clean up
        path.rmdir()


class TestConstants:
    """Tests for module constants."""
    
    def test_datasets_dict(self):
        """Test DATASETS constant."""
        assert isinstance(DATASETS, dict)
        assert DATASETS == {'enriched': 'dataset_stock_hopital_ENRICHI.csv'}
    
    def test_regressors_list(self):
        """Test REGRESSORS constant."""
        assert isinstance(REGRESSORS, list)
        
        expected_regressors = [
            'temperature',
            'taux_occupation',
            'nb_patients_jour'
        ]
        
        for reg in expected_regressors:
            assert reg in REGRESSORS


class TestIntegration:
    """Integration tests for configuration module."""
    
    def test_settings_with_paths(self):
        """Test that settings work with path functions."""
        settings = Settings()
        
        # Should be able to get dataset path
        path = get_dataset_path(DATASETS['enriched'])
        assert path.suffix == ".csv"
    
    def test_config_files_valid_yaml(self, config_dir):
        """Test that all YAML files in config are valid."""
        for yaml_file in config_dir.glob("*.yaml"):
            with open(yaml_file, 'r', encoding='utf-8') as f:
                try:
                    config = yaml.safe_load(f)
                    assert config is not None or config == {}
                except yaml.YAMLError as e:
                    pytest.fail(f"Invalid YAML in {yaml_file}: {e}")
