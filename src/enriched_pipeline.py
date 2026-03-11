"""Notebook-aligned enriched forecasting pipeline.

This module mirrors the business logic used in Analyse_Mont_Vert_ENRICHI.ipynb
so the API can return comparable model metrics, future predictions, and order
recommendations for the enriched dataset.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.metrics import calculate_metrics
from src.model import check_prophet_available

try:
    from prophet import Prophet
except ImportError:
    Prophet = None


ENRICHED_REGRESSORS = [
    "temperature",
    "taux_occupation",
    "nb_patients",
    "epidemie_grippe",
]

CONTINUOUS_REGRESSORS = [
    "temperature",
    "taux_occupation",
    "nb_patients",
]

BINARY_REGRESSORS = [
    "epidemie_grippe",
    "jour_ferie",
    "covid_impact",
]

MANUAL_CHANGEPOINTS = [
    "2020-03-15",
    "2020-11-01",
    "2021-05-01",
    "2022-01-01",
    "2023-09-01",
]

DEFAULT_ORDER_HORIZON_DAYS = 7
DEFAULT_SAFETY_COVERAGE_DAYS = 2


def _ensure_required_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing_columns = [column for column in columns if column not in df.columns]
    if missing_columns:
        raise ValueError(
            "Missing required columns for enriched notebook pipeline: "
            + ", ".join(missing_columns)
        )


def _build_holidays(daily: pd.DataFrame) -> pd.DataFrame:
    holidays_jf = daily[daily["jour_ferie"] == 1][["date"]].drop_duplicates().copy()
    holidays_jf.columns = ["ds"]
    holidays_jf["holiday"] = "jour_ferie"
    holidays_jf["lower_window"] = 0
    holidays_jf["upper_window"] = 0

    holidays_covid = daily[daily["covid_impact"] == 1][["date"]].drop_duplicates().copy()
    holidays_covid.columns = ["ds"]
    holidays_covid["holiday"] = "covid_19"
    holidays_covid["lower_window"] = 0
    holidays_covid["upper_window"] = 0

    return pd.concat([holidays_jf, holidays_covid], ignore_index=True)


def _build_prophet_base_model(holidays: pd.DataFrame) -> Any:
    if not check_prophet_available() or Prophet is None:
        raise ImportError("Prophet is not installed. Run: pip install prophet")

    model = Prophet(
        holidays=holidays,
        holidays_prior_scale=10.0,
        yearly_seasonality=20,
        weekly_seasonality=5,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        seasonality_prior_scale=10.0,
        changepoints=MANUAL_CHANGEPOINTS,
        changepoint_prior_scale=0.5,
        changepoint_range=0.9,
        interval_width=0.85,
        growth="linear",
        mcmc_samples=0,
    )

    return model


def _create_notebook_evaluation_model(holidays: pd.DataFrame) -> Any:
    model = _build_prophet_base_model(holidays)

    model.add_regressor("temperature", prior_scale=0.5, standardize=True, mode="additive")
    model.add_regressor("taux_occupation", prior_scale=1.0, standardize=True, mode="additive")
    model.add_regressor("nb_patients", prior_scale=0.5, standardize=True, mode="additive")
    model.add_regressor("epidemie_grippe", prior_scale=0.5, standardize=False, mode="additive")
    return model


def _create_notebook_forecast_model(holidays: pd.DataFrame) -> Any:
    model = _build_prophet_base_model(holidays)

    model.add_regressor("temperature", prior_scale=0.5, standardize=True)
    model.add_regressor("taux_occupation", prior_scale=1.0, standardize=True)
    model.add_regressor("nb_patients", prior_scale=0.5, standardize=True)
    model.add_regressor("epidemie_grippe", prior_scale=0.5, standardize=False)
    return model


def prepare_notebook_enriched_data(df: pd.DataFrame, product_name: str) -> Dict[str, pd.DataFrame]:
    """Prepare the enriched dataset exactly like the notebook."""
    _ensure_required_columns(
        df,
        [
            "date",
            "nom_produit",
            "type_sortie",
            "quantite",
            "date_expiration",
            "stock_theorique",
            *CONTINUOUS_REGRESSORS,
            *BINARY_REGRESSORS,
        ],
    )

    working_df = df.copy()
    working_df["date"] = pd.to_datetime(working_df["date"])
    working_df["date_expiration"] = pd.to_datetime(working_df["date_expiration"])

    produit_df = working_df[
        (working_df["nom_produit"] == product_name)
        & (working_df["type_sortie"] == "CONSOMMATION")
    ].copy()

    if produit_df.empty:
        raise ValueError(f"No data found for product '{product_name}'")

    daily = produit_df.groupby("date").agg(
        {
            "quantite": "sum",
            "temperature": "mean",
            "taux_occupation": "mean",
            "nb_patients": "mean",
            "epidemie_grippe": "max",
            "jour_ferie": "max",
            "covid_impact": "max",
        }
    ).reset_index()

    date_range = pd.date_range(
        start=daily["date"].min(),
        end=daily["date"].max(),
        freq="D",
    )
    full_dates = pd.DataFrame({"date": date_range})
    daily = full_dates.merge(daily, on="date", how="left")

    daily["quantite"] = daily["quantite"].fillna(0)
    for column in CONTINUOUS_REGRESSORS:
        daily[column] = daily[column].fillna(daily[column].mean())
    for column in BINARY_REGRESSORS:
        daily[column] = daily[column].fillna(0)

    prophet_df = daily.rename(columns={"date": "ds", "quantite": "y"}).copy()
    return {
        "produit_df": produit_df,
        "daily": daily,
        "prophet_df": prophet_df,
        "holidays": _build_holidays(daily),
    }


def _build_future_frame(
    prophet_df: pd.DataFrame,
    periods: int,
    start_date: Optional[str] = None,
) -> pd.DataFrame:
    if start_date:
        future_dates = pd.date_range(start=pd.to_datetime(start_date), periods=periods, freq="D")
    else:
        future_dates = pd.date_range(
            start=prophet_df["ds"].max() + timedelta(days=1),
            periods=periods,
            freq="D",
        )

    future = pd.DataFrame({"ds": future_dates})
    future = future.merge(
        prophet_df[["ds", *ENRICHED_REGRESSORS]],
        on="ds",
        how="left",
    )

    for column in CONTINUOUS_REGRESSORS:
        future[column] = future[column].fillna(prophet_df[column].mean())

    future["epidemie_grippe"] = future["epidemie_grippe"].fillna(
        future["ds"].dt.month.isin([1, 2, 3]).astype(int)
    )
    return future


def build_order_recommendation(
    produit_df: pd.DataFrame,
    predictions_futures: pd.DataFrame,
    horizon_days: int = DEFAULT_ORDER_HORIZON_DAYS,
    safety_coverage_days: int = DEFAULT_SAFETY_COVERAGE_DAYS,
) -> Dict[str, Any]:
    """Reproduce the notebook order recommendation step."""
    date_debut_prevision = predictions_futures["ds"].min().normalize()

    stock_snapshot = produit_df[produit_df["date"] == produit_df["date"].max()].copy()
    stock_disponible = stock_snapshot.loc[
        stock_snapshot["date_expiration"] >= date_debut_prevision,
        "stock_theorique",
    ].sum()

    arrivages_planifies = 0.0

    predictions_commande = predictions_futures.head(horizon_days).copy()
    predictions_commande["yhat"] = predictions_commande["yhat"].clip(lower=0)
    predictions_commande["yhat_upper"] = predictions_commande["yhat_upper"].clip(lower=0)

    consommation_prevue_commande = predictions_commande["yhat"].sum()
    stock_securite_commande = predictions_commande["yhat"].mean() * safety_coverage_days
    quantite_a_commander = max(
        consommation_prevue_commande + stock_securite_commande - stock_disponible - arrivages_planifies,
        0,
    )

    couverture_estimee_jours = (
        stock_disponible / predictions_commande["yhat"].mean()
        if predictions_commande["yhat"].mean() > 0
        else float("inf")
    )

    return {
        "horizon_commande_jours": int(horizon_days),
        "stock_disponible_actuel": float(stock_disponible),
        "arrivages_planifies": float(arrivages_planifies),
        "consommation_prevue_horizon": float(consommation_prevue_commande),
        "stock_securite": float(stock_securite_commande),
        "quantite_a_commander": float(quantite_a_commander),
        "couverture_estimee_jours": (
            float(couverture_estimee_jours)
            if np.isfinite(couverture_estimee_jours)
            else None
        ),
        "hypothese": "Aucun arrivage futur planifié disponible dans le dataset historique",
        "date_debut_prevision": date_debut_prevision,
    }


def run_enriched_notebook_analysis(
    df: pd.DataFrame,
    product_name: str,
    periods: int,
    start_date: Optional[str] = None,
    order_horizon_days: int = DEFAULT_ORDER_HORIZON_DAYS,
    safety_coverage_days: int = DEFAULT_SAFETY_COVERAGE_DAYS,
) -> Dict[str, Any]:
    """Run the notebook-equivalent enriched analysis for one product."""
    prepared = prepare_notebook_enriched_data(df, product_name)
    produit_df = prepared["produit_df"]
    prophet_df = prepared["prophet_df"]
    holidays = prepared["holidays"]

    split_date = prophet_df["ds"].max() - pd.Timedelta(days=365)
    train = prophet_df[prophet_df["ds"] <= split_date].copy()
    test = prophet_df[prophet_df["ds"] > split_date].copy()

    evaluation_model = _create_notebook_evaluation_model(holidays)
    evaluation_model.fit(train)
    predictions_test = evaluation_model.predict(test)

    metrics = calculate_metrics(test["y"].values, predictions_test["yhat"].values)

    forecast_model = _create_notebook_forecast_model(holidays)
    forecast_model.fit(prophet_df)

    future = _build_future_frame(prophet_df, periods=periods, start_date=start_date)
    predictions_futures = forecast_model.predict(future)

    recommendation = build_order_recommendation(
        produit_df=produit_df,
        predictions_futures=predictions_futures,
        horizon_days=order_horizon_days,
        safety_coverage_days=safety_coverage_days,
    )

    prophet_settings = {
        "seasonality_mode": "multiplicative",
        "changepoint_prior_scale": 0.5,
        "interval_width": 0.85,
        "daily_seasonality": False,
        "weekly_seasonality": 5,
        "yearly_seasonality": 20,
        "regressors": ENRICHED_REGRESSORS,
        "holidays": ["jour_ferie", "covid_19"],
        "manual_changepoints": MANUAL_CHANGEPOINTS,
        "order_horizon_days": order_horizon_days,
        "safety_coverage_days": safety_coverage_days,
    }

    return {
        "produit_df": produit_df,
        "daily": prepared["daily"],
        "prophet_df": prophet_df,
        "holidays": holidays,
        "train": train,
        "test": test,
        "predictions_test": predictions_test,
        "predictions_futures": predictions_futures,
        "metrics": metrics,
        "evaluation_model": evaluation_model,
        "forecast_model": forecast_model,
        "recommendation": recommendation,
        "prophet_settings": prophet_settings,
    }