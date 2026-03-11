"""
FastAPI Application
===================

REST API for hospital stock prediction.

Run with:
    uvicorn api.main:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Optional
from datetime import datetime
import sys
from pathlib import Path

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
    AnalysisRequest,
    AnalysisResponse,
    DatasetInfo,
    ProductsResponse,
    ProductInfo,
    HealthResponse,
    ErrorResponse,
    DatasetEnum,
    PriorityEnum
)

from config import (
    settings,
    load_yaml_config,
    get_dataset_path,
    PROJECT_ROOT,
    DATASETS,
    REGRESSORS
)

from src import __version__
from src.data_loader import load_dataset, prepare_prophet_data, train_test_split, get_dataset_info
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
    * **Analyse** - Statistiques et exploration des datasets
    * **Produits** - Gestion des produits configurés
    * **Alertes** - Alertes de stock basées sur les prédictions
    
    ## Datasets disponibles
    
    - `base` - Dataset de base (51k lignes)
    - `realistic` - Dataset réaliste (24k lignes)  
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
        "datasets": list(DATASETS.keys()),
        "regressors": REGRESSORS
    }


# =============================================================================
# Products Endpoints
# =============================================================================

@app.get(
    "/products",
    response_model=ProductsResponse,
    summary="List all products",
    tags=["Products"]
)
async def list_products():
    """Get list of all configured products."""
    try:
        config = load_yaml_config("products")
        products_data = config.get("products", {})
        
        products = []
        for key, prod in products_data.items():
            products.append(ProductInfo(
                id=key,
                name=prod.get("name", key),
                category=prod.get("category", "unknown"),
                dlc_days=prod.get("dlc_days", 0),
                priority=prod.get("priority", "medium"),
                unit=prod.get("unit", "kg"),
                min_stock=prod.get("min_stock"),
                max_stock=prod.get("max_stock"),
                reorder_point=prod.get("reorder_point")
            ))
        
        return ProductsResponse(count=len(products), products=products)
    
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Products configuration not found")


@app.get(
    "/products/{product_id}",
    response_model=ProductInfo,
    summary="Get product details",
    tags=["Products"]
)
async def get_product(product_id: str):
    """Get details for a specific product."""
    try:
        config = load_yaml_config("products")
        products_data = config.get("products", {})
        
        if product_id not in products_data:
            raise HTTPException(status_code=404, detail=f"Product '{product_id}' not found")
        
        prod = products_data[product_id]
        return ProductInfo(
            id=product_id,
            name=prod.get("name", product_id),
            category=prod.get("category", "unknown"),
            dlc_days=prod.get("dlc_days", 0),
            priority=prod.get("priority", "medium"),
            unit=prod.get("unit", "kg"),
            min_stock=prod.get("min_stock"),
            max_stock=prod.get("max_stock"),
            reorder_point=prod.get("reorder_point")
        )
    
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Products configuration not found")


# =============================================================================
# Datasets Endpoints
# =============================================================================

@app.get(
    "/datasets",
    summary="List available datasets",
    tags=["Datasets"]
)
async def list_datasets():
    """Get list of available datasets with their status."""
    datasets_info = []
    
    for key, filename in DATASETS.items():
        path = get_dataset_path(filename)
        datasets_info.append({
            "key": key,
            "filename": filename,
            "exists": path.exists(),
            "path": str(path)
        })
    
    return {"datasets": datasets_info}


@app.get(
    "/datasets/{dataset_key}/info",
    response_model=DatasetInfo,
    summary="Get dataset information",
    tags=["Datasets"]
)
async def get_dataset_info_endpoint(dataset_key: DatasetEnum):
    """Get detailed information about a dataset."""
    if dataset_key.value not in DATASETS:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_key}' not found")
    
    dataset_path = get_dataset_path(DATASETS[dataset_key.value])
    
    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail=f"Dataset file not found: {dataset_path}")
    
    try:
        df = load_dataset(str(dataset_path))
        info = get_dataset_info(df)
        
        # Get unique products
        product_col = next((c for c in ['nom_produit', 'produit', 'product'] if c in df.columns), None)
        products = list(df[product_col].unique()) if product_col else None
        
        # Format date range
        date_range = None
        if 'date_range' in info:
            date_range = {
                "start": str(info['date_range']['start'].date()) if info['date_range']['start'] else None,
                "end": str(info['date_range']['end'].date()) if info['date_range']['end'] else None,
                "days": info['date_range']['days']
            }
        
        return DatasetInfo(
            name=dataset_key.value,
            rows=info['rows'],
            columns=info['columns'],
            memory_mb=round(info['memory_mb'], 2),
            date_range=date_range,
            products=products
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
        404: {"description": "Product or dataset not found"},
        503: {"description": "Prophet not available"}
    }
)
async def predict(request: PredictionRequest):
    """
    Generate consumption predictions for a product.
    
    - **product**: Name of the product to predict
    - **days**: Number of days to predict (1-365)
    - **dataset**: Dataset to use for training
    - **include_regressors**: Include external factors (only for enriched dataset)
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
    dataset_file = DATASETS.get(request.dataset.value)
    if not dataset_file:
        raise HTTPException(status_code=404, detail=f"Dataset '{request.dataset}' not configured")
    
    dataset_path = get_dataset_path(dataset_file)
    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail=f"Dataset file not found")
    
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

        if request.dataset == DatasetEnum.enriched and request.include_regressors:
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
                    dataset_name=request.dataset.value,
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
                dataset=request.dataset.value,
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
                dataset_name=request.dataset.value,
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
            dataset=request.dataset.value,
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
    dataset: DatasetEnum = Query(default=DatasetEnum.enriched),
    enable_mlflow: bool = Query(default=False),
    mlflow_experiment: Optional[str] = Query(default="hospital-stock-api")
):
    """Quick prediction endpoint with URL parameters."""
    request = PredictionRequest(
        product=product,
        days=days,
        dataset=dataset,
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
