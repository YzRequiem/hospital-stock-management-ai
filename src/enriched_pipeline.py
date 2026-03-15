"""Notebook-aligned enriched forecasting pipeline.

This module mirrors the business logic used in Analyse_Mont_Vert_ENRICHI.ipynb
so the API can return comparable model metrics, future predictions, and order
recommendations for the enriched dataset.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any, Dict, Optional
import unicodedata

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
DEFAULT_MIN_DLC_DAYS = 1


def _ensure_required_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing_columns = [column for column in columns if column not in df.columns]
    if missing_columns:
        raise ValueError(
            "Missing required columns for enriched notebook pipeline: "
            + ", ".join(missing_columns)
        )


def _slugify_product_name(name: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(name))
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    return "_".join(ascii_text.lower().split())


def _load_product_config_index() -> Dict[str, Dict[str, Any]]:
    from config import load_yaml_config

    try:
        config = load_yaml_config("products")
    except FileNotFoundError:
        return {}

    indexed_config: Dict[str, Dict[str, Any]] = {}
    for config_key, product_data in config.get("products", {}).items():
        indexed_config[_slugify_product_name(config_key)] = product_data
        indexed_config[_slugify_product_name(product_data.get("name", config_key))] = product_data

    return indexed_config


def infer_product_dlc_days(
    produit_df: pd.DataFrame,
    product_name: Optional[str] = None,
    default_dlc_days: int = DEFAULT_MIN_DLC_DAYS,
) -> int:
    """Infer a product DLC from config first, then from observed expiration delays."""
    if product_name:
        product_config = _load_product_config_index().get(_slugify_product_name(product_name))
        if product_config and product_config.get("dlc_days"):
            return max(int(product_config["dlc_days"]), DEFAULT_MIN_DLC_DAYS)

    if {"date", "date_expiration"}.issubset(produit_df.columns):
        expiration_delta = (
            pd.to_datetime(produit_df["date_expiration"], errors="coerce")
            - pd.to_datetime(produit_df["date"], errors="coerce")
        ).dt.days
        expiration_delta = expiration_delta[expiration_delta.notna() & (expiration_delta > 0)]
        if not expiration_delta.empty:
            return max(int(round(float(expiration_delta.median()))), DEFAULT_MIN_DLC_DAYS)

    return max(int(default_dlc_days), DEFAULT_MIN_DLC_DAYS)


def estimate_current_usable_stock(
    produit_df: pd.DataFrame,
    forecast_start: pd.Timestamp,
) -> float:
    """Estimate stock still consumable at the forecast start date."""
    stock_snapshot = produit_df[produit_df["date"] == produit_df["date"].max()].copy()
    if stock_snapshot.empty or "stock_theorique" not in stock_snapshot.columns:
        return 0.0

    if "date_expiration" in stock_snapshot.columns:
        usable_stock = stock_snapshot.loc[
            pd.to_datetime(stock_snapshot["date_expiration"], errors="coerce") >= forecast_start,
            "stock_theorique",
        ]
    else:
        usable_stock = stock_snapshot["stock_theorique"]

    return float(usable_stock.sum())


def estimate_current_stock_remaining_days(
    produit_df: pd.DataFrame,
    default_days: int,
) -> int:
    """Estimate how many days the current stock remains consumable from the last observed date."""
    stock_snapshot = produit_df[produit_df["date"] == produit_df["date"].max()].copy()
    if stock_snapshot.empty or "date_expiration" not in stock_snapshot.columns:
        return max(int(default_days), DEFAULT_MIN_DLC_DAYS)

    remaining_days = (
        pd.to_datetime(stock_snapshot["date_expiration"], errors="coerce")
        - pd.to_datetime(stock_snapshot["date"], errors="coerce")
    ).dt.days
    remaining_days = remaining_days[remaining_days.notna() & (remaining_days > 0)]

    if remaining_days.empty:
        return max(int(default_days), DEFAULT_MIN_DLC_DAYS)

    return max(int(round(float(remaining_days.median()))), DEFAULT_MIN_DLC_DAYS)


def build_dlc_arrival_schedule(
    produit_df: pd.DataFrame,
    predictions_futures: pd.DataFrame,
    product_name: Optional[str] = None,
    dlc_days: Optional[int] = None,
    current_usable_stock: Optional[float] = None,
    current_stock_remaining_days: Optional[int] = None,
) -> pd.DataFrame:
    """Plan replenishment windows so each arrival covers at most the product DLC."""
    if predictions_futures.empty:
        return pd.DataFrame(
            columns=[
                "product_name",
                "arrival_date",
                "coverage_start",
                "coverage_end",
                "dlc_days",
                "window_days",
                "predicted_consumption_window",
                "current_stock_used",
                "quantity_to_order",
                "remaining_current_stock",
            ]
        )

    forecast = predictions_futures[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    forecast["ds"] = pd.to_datetime(forecast["ds"])
    forecast["yhat"] = forecast["yhat"].clip(lower=0)

    inferred_dlc_days = dlc_days or infer_product_dlc_days(produit_df=produit_df, product_name=product_name)
    inferred_dlc_days = max(int(inferred_dlc_days), DEFAULT_MIN_DLC_DAYS)

    forecast_start = forecast["ds"].min().normalize()
    usable_stock = (
        estimate_current_usable_stock(produit_df, forecast_start)
        if current_usable_stock is None
        else float(current_usable_stock)
    )
    stock_remaining_days = (
        estimate_current_stock_remaining_days(produit_df, inferred_dlc_days)
        if current_stock_remaining_days is None
        else int(current_stock_remaining_days)
    )
    stock_remaining_days = max(stock_remaining_days, DEFAULT_MIN_DLC_DAYS)
    stock_usable_until = forecast_start + pd.Timedelta(days=stock_remaining_days - 1)

    arrival_rows = []
    remaining_current_stock = usable_stock

    for start_idx in range(0, len(forecast), inferred_dlc_days):
        window = forecast.iloc[start_idx : start_idx + inferred_dlc_days].copy()
        if window.empty:
            continue

        coverage_start = window["ds"].min().normalize()
        coverage_end = window["ds"].max().normalize()
        predicted_window = float(window["yhat"].sum())

        stock_eligible_mask = window["ds"].dt.normalize() <= stock_usable_until
        stock_eligible_demand = float(window.loc[stock_eligible_mask, "yhat"].sum())
        stock_used = min(remaining_current_stock, stock_eligible_demand)
        quantity_to_order = max(predicted_window - stock_used, 0.0)
        remaining_current_stock = max(remaining_current_stock - stock_used, 0.0)

        arrival_rows.append(
            {
                "product_name": product_name,
                "arrival_date": coverage_start,
                "coverage_start": coverage_start,
                "coverage_end": coverage_end,
                "dlc_days": inferred_dlc_days,
                "window_days": int(len(window)),
                "predicted_consumption_window": predicted_window,
                "current_stock_used": float(stock_used),
                "quantity_to_order": float(quantity_to_order),
                "remaining_current_stock": float(remaining_current_stock),
            }
        )

    return pd.DataFrame(arrival_rows)


def build_multi_product_arrival_plan(
    df: pd.DataFrame,
    periods: int,
    start_date: Optional[str] = None,
    product_names: Optional[list[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """Build a 28-day DLC-aware arrival plan for multiple products."""
    working_df = df.copy()
    if "nom_produit" not in working_df.columns:
        raise ValueError("The dataset must contain a 'nom_produit' column.")

    available_products = sorted(working_df["nom_produit"].dropna().astype(str).unique())
    selected_products = product_names or available_products

    arrival_plans = []
    product_summaries = []

    for product_name in selected_products:
        analysis = run_enriched_notebook_analysis(
            df=working_df,
            product_name=product_name,
            periods=periods,
            start_date=start_date,
        )

        dlc_days = infer_product_dlc_days(analysis["produit_df"], product_name=product_name)
        arrival_plan = build_dlc_arrival_schedule(
            produit_df=analysis["produit_df"],
            predictions_futures=analysis["predictions_futures"],
            product_name=product_name,
            dlc_days=dlc_days,
        )

        if not arrival_plan.empty:
            arrival_plans.append(arrival_plan)

        product_summaries.append(
            {
                "product_name": product_name,
                "dlc_days": dlc_days,
                "forecast_start": pd.to_datetime(analysis["predictions_futures"]["ds"]).min().normalize(),
                "forecast_end": pd.to_datetime(analysis["predictions_futures"]["ds"]).max().normalize(),
                "predicted_total_28d": float(analysis["predictions_futures"]["yhat"].clip(lower=0).sum()),
                "current_usable_stock": float(analysis["recommendation"]["stock_disponible_actuel"]),
                "recommended_total_to_order": float(arrival_plan["quantity_to_order"].sum()) if not arrival_plan.empty else 0.0,
                "num_arrivals": int(len(arrival_plan)),
                "mape": float(analysis["metrics"].get("mape", np.nan)),
                "mae": float(analysis["metrics"].get("mae", np.nan)),
                "rmse": float(analysis["metrics"].get("rmse", np.nan)),
            }
        )

    arrival_plan_df = (
        pd.concat(arrival_plans, ignore_index=True)
        if arrival_plans
        else pd.DataFrame()
    )
    product_summary_df = pd.DataFrame(product_summaries).sort_values(
        ["recommended_total_to_order", "predicted_total_28d"],
        ascending=[False, False],
    )

    return {
        "arrival_plan": arrival_plan_df,
        "product_summary": product_summary_df,
    }


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


def _build_prophet_base_model(
    holidays: pd.DataFrame,
    history_dates: Optional[pd.Series] = None,
) -> Any:
    if not check_prophet_available() or Prophet is None:
        raise ImportError("Prophet is not installed. Run: pip install prophet")

    changepoints = MANUAL_CHANGEPOINTS
    if history_dates is not None and not history_dates.empty:
        history_dates = pd.to_datetime(history_dates)
        min_history_date = history_dates.min()
        max_history_date = history_dates.max()
        changepoints = [
            changepoint
            for changepoint in MANUAL_CHANGEPOINTS
            if min_history_date < pd.Timestamp(changepoint) < max_history_date
        ]

    model = Prophet(
        holidays=holidays,
        holidays_prior_scale=10.0,
        yearly_seasonality=20,
        weekly_seasonality=5,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        seasonality_prior_scale=10.0,
        changepoints=changepoints,
        changepoint_prior_scale=0.5,
        changepoint_range=0.9,
        interval_width=0.85,
        growth="linear",
        mcmc_samples=0,
    )

    return model


def _create_notebook_evaluation_model(holidays: pd.DataFrame, history_dates: pd.Series) -> Any:
    model = _build_prophet_base_model(holidays, history_dates=history_dates)

    model.add_regressor("temperature", prior_scale=0.5, standardize=True, mode="additive")
    model.add_regressor("taux_occupation", prior_scale=1.0, standardize=True, mode="additive")
    model.add_regressor("nb_patients", prior_scale=0.5, standardize=True, mode="additive")
    model.add_regressor("epidemie_grippe", prior_scale=0.5, standardize=False, mode="additive")
    return model


def _create_notebook_forecast_model(holidays: pd.DataFrame, history_dates: pd.Series) -> Any:
    model = _build_prophet_base_model(holidays, history_dates=history_dates)

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

    if len(train) < 2 or test.empty:
        split_index = min(max(int(len(prophet_df) * 0.8), 2), len(prophet_df) - 1)
        train = prophet_df.iloc[:split_index].copy()
        test = prophet_df.iloc[split_index:].copy()

    evaluation_model = _create_notebook_evaluation_model(holidays, history_dates=train["ds"])
    evaluation_model.fit(train)
    predictions_test = evaluation_model.predict(test)

    metrics = calculate_metrics(test["y"].values, predictions_test["yhat"].values)

    forecast_model = _create_notebook_forecast_model(holidays, history_dates=prophet_df["ds"])
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