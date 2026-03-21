"""
FastAPI Application
===================

REST API for hospital stock prediction.

Run with:
    uvicorn api.main:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Any, Dict, Optional
from datetime import datetime
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.models import (
    PredictionRequest,
    PredictionResponse,
    PredictionPoint,
    RecommendationResponse,
    MetricsResponse,
    QualityAssessment,
    TrackingResponse,
    ProductsResponse,
    ProductInfo,
    HealthResponse,
    PriorityEnum
)

from config import (
    settings,
    load_yaml_config,
    get_dataset_path,
    PROJECT_ROOT,
    REGRESSORS
)

from src import __version__
from src.data_loader import load_dataset, prepare_prophet_data, train_test_split
from src.enriched_pipeline import run_enriched_notebook_analysis
from src.metrics import calculate_metrics, interpret_mape
from src.model import check_prophet_available, model_summary
from src.mlflow_utils import (
    check_mlflow_available,
    get_default_tracking_uri,
    log_prediction_run,
    normalize_tracking_uri,
)

# =============================================================================
# Application Setup
# =============================================================================

app = FastAPI(
    title="Hospital Stock Prediction API",
    description="""
    🏥 **Clinique du Mont Vert** - Stock Management AI
    
    API REST pour la prédiction de consommation de stock hospitalier
    utilisant Prophet pour les prévisions de séries temporelles.
    
    ## Fonctionnalités
    
    * **Prédictions** - Prévisions de consommation pour les produits alimentaires
    * **Analyse** - Statistiques et exploration du dataset enrichi
    * **Produits** - Gestion des produits configurés
    * **Alertes** - Alertes de stock basées sur les prédictions
    
    ## Dataset disponible

    - `enriched` - Dataset enrichi avec régresseurs (85k lignes)
    """,
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Health & Info Endpoints
# =============================================================================

@app.get(
    "/",
    summary="Root endpoint",
    response_model=dict
)
async def root():
    """Welcome message and API info."""
    return {
        "message": "🏥 Hospital Stock Prediction API",
        "version": __version__,
        "docs": "/docs",
        "health": "/health"
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["System"]
)
async def health_check():
    """Check API health and dependencies."""
    return HealthResponse(
        status="healthy",
        version=__version__,
        prophet_available=check_prophet_available(),
        timestamp=datetime.now()
    )


@app.get(
    "/info",
    summary="API information",
    tags=["System"]
)
async def api_info():
    """Get detailed API information."""
    return {
        "name": "Hospital Stock Prediction API",
        "version": __version__,
        "clinic": "Clinique du Mont Vert",
        "author": "Master EISI",
        "prophet_available": check_prophet_available(),
        "dataset": "enriched",
        "regressors": REGRESSORS
    }


# =============================================================================
# Products Endpoints
# =============================================================================


def _slugify_product_name(name: str) -> str:
    """Convert a product name into a stable API identifier."""
    normalized = unicodedata.normalize("NFKD", name)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", ascii_text.lower()).strip("_")
    return slug or "unknown_product"


def _safe_mode(series):
    """Return the most frequent non-null value from a Series."""
    non_null = series.dropna()
    if non_null.empty:
        return None
    modes = non_null.mode()
    if modes.empty:
        return None
    return modes.iloc[0]


def _infer_priority_from_dlc(dlc_days: int) -> str:
    """Infer a priority level from shelf life when no config exists."""
    if dlc_days and dlc_days <= 7:
        return "high"
    if dlc_days and dlc_days <= 30:
        return "medium"
    return "low"


def _load_product_config_index() -> Dict[str, Dict[str, Any]]:
    """Load configured product metadata indexed by stable slug."""
    try:
        config = load_yaml_config("products")
    except FileNotFoundError:
        return {}

    products_data = config.get("products", {})
    indexed_config: Dict[str, Dict[str, Any]] = {}

    for config_key, product_data in products_data.items():
        name = product_data.get("name", config_key)
        indexed_config[_slugify_product_name(config_key)] = product_data
        indexed_config[_slugify_product_name(name)] = product_data

    return indexed_config


def _build_products_from_enriched_dataset() -> list[ProductInfo]:
    """Build product metadata from the enriched CSV dataset."""
    dataset_path = get_dataset_path("dataset_stock_hopital_ENRICHI.csv")
    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail="Enriched dataset file not found")

    df = load_dataset(str(dataset_path))

    if 'nom_produit' in df.columns:
        product_col = 'nom_produit'
    elif 'produit' in df.columns:
        product_col = 'produit'
    else:
        raise HTTPException(status_code=500, detail="No product column found in dataset")

    product_config = _load_product_config_index()
    products: list[ProductInfo] = []

    for product_name, product_df in df.groupby(product_col, dropna=True):
        if not isinstance(product_name, str) or not product_name.strip():
            continue

        product_slug = _slugify_product_name(product_name)
        config_data = product_config.get(product_slug, {})

        inferred_category = _safe_mode(product_df['type_produit']) if 'type_produit' in product_df.columns else None
        inferred_unit = _safe_mode(product_df['unite']) if 'unite' in product_df.columns else None

        inferred_dlc_days = None
        if 'date' in product_df.columns and 'date_expiration' in product_df.columns:
            date_values = pd.to_datetime(product_df['date'], errors='coerce')
            expiration_values = pd.to_datetime(product_df['date_expiration'], errors='coerce')
            delta_days = (expiration_values - date_values).dt.days
            delta_days = delta_days[delta_days.notna() & (delta_days >= 0)]
            if not delta_days.empty:
                inferred_dlc_days = int(round(float(delta_days.median())))

        dlc_days = config_data.get('dlc_days')
        if dlc_days is None:
            dlc_days = inferred_dlc_days or 0

        priority = config_data.get('priority') or _infer_priority_from_dlc(int(dlc_days))

        products.append(ProductInfo(
            id=product_slug,
            name=product_name,
            category=config_data.get('category') or inferred_category or 'unknown',
            dlc_days=int(dlc_days),
            priority=priority,
            unit=config_data.get('unit') or inferred_unit or 'kg',
            min_stock=config_data.get('min_stock'),
            max_stock=config_data.get('max_stock'),
            reorder_point=config_data.get('reorder_point')
        ))

    products.sort(key=lambda product: product.name.lower())
    return products

@app.get(
    "/products",
    response_model=ProductsResponse,
    summary="List all products",
    tags=["Products"]
)
async def list_products():
    """Get products actually present in the enriched CSV dataset."""
    products = _build_products_from_enriched_dataset()
    return ProductsResponse(count=len(products), products=products)


@app.get(
    "/products/{product_id}",
    response_model=ProductInfo,
    summary="Get product details",
    tags=["Products"]
)
async def get_product(product_id: str):
    """Get details for a specific dataset product."""
    normalized_product_id = _slugify_product_name(product_id)
    products = _build_products_from_enriched_dataset()

    for product in products:
        if product.id == normalized_product_id:
            return product

    raise HTTPException(status_code=404, detail=f"Product '{product_id}' not found")

# =============================================================================
# Prediction Endpoints
# =============================================================================

@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Generate predictions",
    tags=["Predictions"],
    responses={
        200: {"description": "Successful prediction"},
        400: {"description": "Invalid request"},
        404: {"description": "Product not found"},
        503: {"description": "Prophet not available"}
    }
)
async def predict(request: PredictionRequest):
    """
    Generate consumption predictions for a product.
    
    - **product**: Name of the product to predict
    - **days**: Number of days to predict (1-365)
    - **include_regressors**: Include external factors from the enriched dataset
    """
    # Check Prophet availability
    if not check_prophet_available():
        raise HTTPException(
            status_code=503,
            detail="Prophet is not installed. Please install with: pip install prophet"
        )
    
    # Import Prophet-dependent modules
    from src.model import train_prophet_model, predict as model_predict, predict_future
    
    # Get dataset path
    dataset_path = get_dataset_path("dataset_stock_hopital_ENRICHI.csv")
    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail="Enriched dataset file not found")
    
    try:
        tracking_uri = normalize_tracking_uri(request.mlflow_tracking_uri, PROJECT_ROOT)
        tracking_info = TrackingResponse(
            enabled=request.enable_mlflow,
            logged=False,
            experiment_name=request.mlflow_experiment if request.enable_mlflow else None,
            tracking_uri=tracking_uri if request.enable_mlflow else None,
        )

        # Load and prepare data
        df = load_dataset(str(dataset_path))

        start_date_str = str(request.start_date) if request.start_date else None
        recommendation = None

        if request.include_regressors:
            analysis = run_enriched_notebook_analysis(
                df=df,
                product_name=request.product,
                periods=request.days,
                start_date=start_date_str,
            )

            train = analysis['train']
            test = analysis['test']
            predictions_test = analysis['predictions_test']
            future_preds = analysis['predictions_futures']
            metrics = analysis['metrics']
            model = analysis['forecast_model']
            recommendation = RecommendationResponse(**analysis['recommendation'])
            quality_level, quality_desc = interpret_mape(metrics['mape'])

            if request.enable_mlflow and check_mlflow_available():
                tracked = log_prediction_run(
                    product_name=request.product,
                    dataset_name='enriched',
                    horizon_days=request.days,
                    dataset_path=dataset_path,
                    train_df=train,
                    test_df=test,
                    test_predictions=predictions_test[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy(),
                    future_predictions=future_preds[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy(),
                    metrics=metrics,
                    prophet_settings=analysis['prophet_settings'],
                    model_details=model_summary(model),
                    experiment_name=request.mlflow_experiment or 'hospital-stock-api',
                    tracking_uri=tracking_uri,
                    saved_results_dir=None,
                    model=model,
                    run_tags={
                        'source': 'api',
                        'pipeline': 'notebook-aligned-enriched',
                        'include_regressors': request.include_regressors,
                        'start_date': start_date_str,
                    },
                )

                tracking_info.logged = tracked
                if tracked:
                    import mlflow

                    active_run = mlflow.last_active_run()
                    if active_run is not None:
                        tracking_info.run_id = active_run.info.run_id
                        tracking_info.run_name = active_run.data.tags.get('mlflow.runName')
            elif request.enable_mlflow:
                tracking_info.logged = False

            predictions = []
            for _, row in future_preds.iterrows():
                predictions.append(PredictionPoint(
                    date=row['ds'].date(),
                    predicted=round(row['yhat'], 2),
                    lower_bound=round(row['yhat_lower'], 2),
                    upper_bound=round(row['yhat_upper'], 2)
                ))

            return PredictionResponse(
                product=request.product,
                dataset='enriched',
                train_days=len(train),
                prediction_days=request.days,
                metrics=MetricsResponse(
                    mae=round(metrics['mae'], 2),
                    mape=round(metrics['mape'], 2),
                    rmse=round(metrics['rmse'], 2),
                    r2=round(metrics['r2'], 4)
                ),
                quality=QualityAssessment(
                    level=quality_level,
                    description=quality_desc
                ),
                predictions=predictions,
                recommendation=recommendation,
                tracking=tracking_info,
                generated_at=datetime.now()
            )
        
        # Filter only CONSUMPTION outputs (like in the notebook)
        # Exclude: destructions, entries (arrivals)
        if 'type_sortie' in df.columns:
            df = df[df['type_sortie'] == 'CONSOMMATION'].copy()
        elif 'type_operation' in df.columns:
            df = df[df['type_operation'] == 'SORTIE'].copy()
        
        # Detect target column (quantite_consommee or quantite)
        if 'quantite_consommee' in df.columns:
            target_col = 'quantite_consommee'
        elif 'quantite' in df.columns:
            target_col = 'quantite'
        else:
            raise HTTPException(status_code=500, detail="No quantity column found in dataset")
        
        # Detect product column
        if 'nom_produit' in df.columns:
            product_col = 'nom_produit'
        elif 'produit' in df.columns:
            product_col = 'produit'
        else:
            raise HTTPException(status_code=500, detail="No product column found in dataset")
        
        prophet_df = prepare_prophet_data(
            df,
            target_column=target_col,
            product_filter=request.product,
            product_column=product_col
        )
        
        if len(prophet_df) == 0:
            raise HTTPException(
                status_code=404,
                detail=f"No data found for product '{request.product}'"
            )
        
        # Split data
        train, test = train_test_split(prophet_df, test_ratio=0.2)
        
        # Train model
        model = train_prophet_model(
            train,
            daily_seasonality=settings.prophet.daily_seasonality,
            weekly_seasonality=settings.prophet.weekly_seasonality,
            yearly_seasonality=settings.prophet.yearly_seasonality,
            seasonality_mode=settings.prophet.seasonality_mode,
            changepoint_prior_scale=settings.prophet.changepoint_prior_scale,
            verbose=False
        )
        
        # Evaluate
        predictions_test = model_predict(model, test)
        y_true = test['y'].values
        y_pred = predictions_test['yhat'].values
        
        metrics = calculate_metrics(y_true, y_pred)
        quality_level, quality_desc = interpret_mape(metrics['mape'])
        
        # Future predictions - use start_date if provided, otherwise today
        future_preds = predict_future(model, periods=request.days, start_date=start_date_str)

        if request.enable_mlflow and check_mlflow_available():
            tracked = log_prediction_run(
                product_name=request.product,
                dataset_name='enriched',
                horizon_days=request.days,
                dataset_path=dataset_path,
                train_df=train,
                test_df=test,
                test_predictions=predictions_test[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy(),
                future_predictions=future_preds[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy(),
                metrics=metrics,
                prophet_settings={
                    **settings.to_dict().get('prophet', {}),
                    'include_regressors': request.include_regressors,
                },
                model_details=model_summary(model),
                experiment_name=request.mlflow_experiment or 'hospital-stock-api',
                tracking_uri=tracking_uri,
                saved_results_dir=None,
                model=model,
                run_tags={
                    'source': 'api',
                    'include_regressors': request.include_regressors,
                    'start_date': start_date_str,
                },
            )

            tracking_info.logged = tracked
            if tracked:
                import mlflow

                active_run = mlflow.last_active_run()
                if active_run is not None:
                    tracking_info.run_id = active_run.info.run_id
                    tracking_info.run_name = active_run.data.tags.get('mlflow.runName')
        elif request.enable_mlflow:
            tracking_info.logged = False
        
        # Format predictions
        predictions = []
        for _, row in future_preds.iterrows():
            predictions.append(PredictionPoint(
                date=row['ds'].date(),
                predicted=round(row['yhat'], 2),
                lower_bound=round(row['yhat_lower'], 2),
                upper_bound=round(row['yhat_upper'], 2)
            ))
        
        return PredictionResponse(
            product=request.product,
            dataset='enriched',
            train_days=len(train),
            prediction_days=request.days,
            metrics=MetricsResponse(
                mae=round(metrics['mae'], 2),
                mape=round(metrics['mape'], 2),
                rmse=round(metrics['rmse'], 2),
                r2=round(metrics['r2'], 4)
            ),
            quality=QualityAssessment(
                level=quality_level,
                description=quality_desc
            ),
            predictions=predictions,
            recommendation=recommendation,
            tracking=tracking_info,
            generated_at=datetime.now()
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/predict/{product}",
    response_model=PredictionResponse,
    summary="Quick prediction",
    tags=["Predictions"]
)
async def quick_predict(
    product: str,
    days: int = Query(default=30, ge=1, le=365),
    enable_mlflow: bool = Query(default=False),
    mlflow_experiment: Optional[str] = Query(default="hospital-stock-api")
):
    """Quick prediction endpoint with URL parameters."""
    request = PredictionRequest(
        product=product,
        days=days,
        enable_mlflow=enable_mlflow,
        mlflow_experiment=mlflow_experiment,
    )
    return await predict(request)


# =============================================================================
# Error Handlers
# =============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )


# =============================================================================
# Run with uvicorn
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
