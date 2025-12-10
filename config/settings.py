"""
Configuration Settings Module
=============================

Centralized configuration management for the hospital stock prediction system.
Supports loading from YAML files and environment variables.
"""

import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"


@dataclass
class ProphetConfig:
    """Prophet model configuration."""
    daily_seasonality: bool = False
    weekly_seasonality: bool = True
    yearly_seasonality: bool = True
    seasonality_mode: str = "additive"
    interval_width: float = 0.85
    changepoint_prior_scale: float = 0.05
    

@dataclass
class DataConfig:
    """Data processing configuration."""
    date_column: str = "date"
    target_column: str = "quantite_consommee"
    product_column: str = "produit"
    encoding: str = "utf-8"
    test_ratio: float = 0.2
    fill_missing_dates: bool = True


@dataclass
class OutputConfig:
    """Output and results configuration."""
    save_plots: bool = True
    plot_format: str = "png"
    plot_dpi: int = 150
    generate_report: bool = True
    results_subdir_format: str = "analyse-{product}-{date}"


@dataclass 
class Settings:
    """Main settings container."""
    prophet: ProphetConfig = field(default_factory=ProphetConfig)
    data: DataConfig = field(default_factory=DataConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    
    # Paths
    project_root: Path = PROJECT_ROOT
    config_dir: Path = CONFIG_DIR
    data_dir: Path = DATA_DIR
    results_dir: Path = RESULTS_DIR
    
    @classmethod
    def from_yaml(cls, filepath: str) -> "Settings":
        """Load settings from a YAML file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        settings = cls()
        
        if 'prophet' in config:
            settings.prophet = ProphetConfig(**config['prophet'])
        if 'data' in config:
            settings.data = DataConfig(**config['data'])
        if 'output' in config:
            settings.output = OutputConfig(**config['output'])
            
        return settings
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary."""
        return {
            "prophet": {
                "daily_seasonality": self.prophet.daily_seasonality,
                "weekly_seasonality": self.prophet.weekly_seasonality,
                "yearly_seasonality": self.prophet.yearly_seasonality,
                "seasonality_mode": self.prophet.seasonality_mode,
                "interval_width": self.prophet.interval_width,
                "changepoint_prior_scale": self.prophet.changepoint_prior_scale,
            },
            "data": {
                "date_column": self.data.date_column,
                "target_column": self.data.target_column,
                "product_column": self.data.product_column,
                "encoding": self.data.encoding,
                "test_ratio": self.data.test_ratio,
                "fill_missing_dates": self.data.fill_missing_dates,
            },
            "output": {
                "save_plots": self.output.save_plots,
                "plot_format": self.output.plot_format,
                "plot_dpi": self.output.plot_dpi,
                "generate_report": self.output.generate_report,
                "results_subdir_format": self.output.results_subdir_format,
            }
        }


def load_yaml_config(filename: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file from the config directory.
    
    Args:
        filename: Name of the YAML file (with or without .yaml extension)
        
    Returns:
        Dictionary with configuration values
    """
    if not filename.endswith(('.yaml', '.yml')):
        filename = f"{filename}.yaml"
    
    filepath = CONFIG_DIR / filename
    
    if not filepath.exists():
        raise FileNotFoundError(f"Configuration file not found: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def get_dataset_path(dataset_name: str) -> Path:
    """
    Get the full path to a dataset file.
    
    Args:
        dataset_name: Name of the dataset (with or without .csv extension)
        
    Returns:
        Full path to the dataset
    """
    if not dataset_name.endswith('.csv'):
        dataset_name = f"{dataset_name}.csv"
    
    return DATA_DIR / dataset_name


def get_results_path(subdir: Optional[str] = None) -> Path:
    """
    Get the results directory path, optionally creating a subdirectory.
    
    Args:
        subdir: Optional subdirectory name
        
    Returns:
        Path to results directory
    """
    path = RESULTS_DIR
    if subdir:
        path = path / subdir
    
    path.mkdir(parents=True, exist_ok=True)
    return path


def create_results_directory(product_name: str) -> Path:
    """
    Create a timestamped results directory for a product analysis.
    
    Args:
        product_name: Name of the product being analyzed
        
    Returns:
        Path to the created directory
    """
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_name = product_name.lower().replace(" ", "_").replace("é", "e")
    dirname = f"analyse-{clean_name}-{timestamp}"
    
    path = RESULTS_DIR / dirname
    path.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Dossier créé: {path}")
    return path


# Default settings instance
settings = Settings()


# Available datasets
DATASETS = {
    "base": "stock_hospital_base.csv",
    "realistic": "stock_hospital_realiste.csv",
    "enriched": "stock_hospital_enrichi.csv"
}


# Available regressors for enriched dataset
REGRESSORS = [
    "temperature",
    "taux_occupation", 
    "nb_patients_jour",
    "epidemie_active",
    "jour_ferie",
    "periode_covid"
]
