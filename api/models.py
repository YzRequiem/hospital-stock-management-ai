"""
Pydantic Models for API
=======================

Data validation models for the REST API.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import date, datetime
from enum import Enum


class DatasetEnum(str, Enum):
    """Available datasets."""
    base = "base"
    realistic = "realistic"
    enriched = "enriched"


class PriorityEnum(str, Enum):
    """Product priority levels."""
    high = "high"
    medium = "medium"
    low = "low"


# =============================================================================
# Request Models
# =============================================================================

class PredictionRequest(BaseModel):
    """Request model for predictions."""
    product: str = Field(
        ..., 
        description="Product name to predict",
        example="Poulet Frais"
    )
    days: int = Field(
        default=30,
        ge=1,
        le=365,
        description="Number of days to predict"
    )
    dataset: DatasetEnum = Field(
        default=DatasetEnum.enriched,
        description="Dataset to use for training"
    )
    include_regressors: bool = Field(
        default=False,
        description="Include external regressors (enriched dataset only)"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "product": "Poulet Frais",
                "days": 30,
                "dataset": "enriched",
                "include_regressors": False
            }
        }


class AnalysisRequest(BaseModel):
    """Request model for dataset analysis."""
    dataset: DatasetEnum = Field(
        default=DatasetEnum.enriched,
        description="Dataset to analyze"
    )
    product: Optional[str] = Field(
        default=None,
        description="Optional product filter"
    )


class TrainRequest(BaseModel):
    """Request model for model training."""
    product: str = Field(..., description="Product to train model for")
    dataset: DatasetEnum = Field(default=DatasetEnum.enriched)
    
    # Prophet parameters
    daily_seasonality: bool = Field(default=False)
    weekly_seasonality: bool = Field(default=True)
    yearly_seasonality: bool = Field(default=True)
    seasonality_mode: str = Field(default="additive")
    changepoint_prior_scale: float = Field(default=0.05, ge=0.001, le=0.5)


# =============================================================================
# Response Models
# =============================================================================

class PredictionPoint(BaseModel):
    """Single prediction point."""
    date: date
    predicted: float = Field(..., description="Predicted value in kg")
    lower_bound: float = Field(..., description="Lower confidence bound")
    upper_bound: float = Field(..., description="Upper confidence bound")


class MetricsResponse(BaseModel):
    """Model performance metrics."""
    mae: float = Field(..., description="Mean Absolute Error")
    mape: float = Field(..., description="Mean Absolute Percentage Error")
    rmse: float = Field(..., description="Root Mean Square Error")
    r2: float = Field(..., description="R² coefficient")
    
    class Config:
        schema_extra = {
            "example": {
                "mae": 5.23,
                "mape": 12.45,
                "rmse": 7.89,
                "r2": 0.85
            }
        }


class QualityAssessment(BaseModel):
    """Quality assessment of predictions."""
    level: str = Field(..., description="Quality level: excellent, very_good, good, acceptable, poor")
    description: str = Field(..., description="Human-readable description")


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    product: str
    dataset: str
    train_days: int = Field(..., description="Number of training days")
    prediction_days: int = Field(..., description="Number of predicted days")
    metrics: MetricsResponse
    quality: QualityAssessment
    predictions: List[PredictionPoint]
    generated_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        schema_extra = {
            "example": {
                "product": "Poulet Frais",
                "dataset": "enriched",
                "train_days": 1460,
                "prediction_days": 30,
                "metrics": {
                    "mae": 5.23,
                    "mape": 12.45,
                    "rmse": 7.89,
                    "r2": 0.85
                },
                "quality": {
                    "level": "very_good",
                    "description": "✅ Très bonne précision (MAPE < 15%)"
                },
                "predictions": [
                    {"date": "2025-01-01", "predicted": 45.5, "lower_bound": 38.2, "upper_bound": 52.8}
                ],
                "generated_at": "2025-12-10T14:30:00"
            }
        }


class DatasetInfo(BaseModel):
    """Dataset information."""
    name: str
    rows: int
    columns: int
    memory_mb: float
    date_range: Optional[Dict[str, Any]] = None
    products: Optional[List[str]] = None


class AnalysisResponse(BaseModel):
    """Response model for dataset analysis."""
    dataset: DatasetInfo
    column_stats: Dict[str, Dict[str, Any]]
    numeric_summary: Dict[str, Dict[str, float]]


class ProductInfo(BaseModel):
    """Product information from configuration."""
    id: str
    name: str
    category: str
    dlc_days: int
    priority: PriorityEnum
    unit: str = "kg"
    min_stock: Optional[float] = None
    max_stock: Optional[float] = None
    reorder_point: Optional[float] = None


class ProductsResponse(BaseModel):
    """Response model for products list."""
    count: int
    products: List[ProductInfo]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    version: str
    prophet_available: bool
    timestamp: datetime = Field(default_factory=datetime.now)


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


# =============================================================================
# Stock Alert Models
# =============================================================================

class StockAlert(BaseModel):
    """Stock alert information."""
    product: str
    current_stock: float
    predicted_consumption: float
    days_until_reorder: int
    alert_level: str = Field(..., description="critical, warning, normal")
    recommendation: str


class StockAlertsResponse(BaseModel):
    """Response model for stock alerts."""
    date: date
    alerts: List[StockAlert]
    critical_count: int
    warning_count: int
