"""Tests for the notebook-aligned enriched pipeline."""

from pathlib import Path

import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.enriched_pipeline import build_dlc_arrival_schedule, run_enriched_notebook_analysis


class TestEnrichedPipeline:
    """Focused coverage for the notebook-aligned enriched analysis."""

    def test_returns_predictions_and_recommendation(self, sample_enriched_data):
        sample_enriched_data = sample_enriched_data.rename(
            columns={
                'produit': 'nom_produit',
                'quantite_consommee': 'quantite',
                'nb_patients_jour': 'nb_patients',
                'epidemie_active': 'epidemie_grippe',
                'periode_covid': 'covid_impact',
            }
        )
        sample_enriched_data['type_sortie'] = 'CONSOMMATION'
        sample_enriched_data['date_expiration'] = sample_enriched_data['date'] + pd.Timedelta(days=3)
        sample_enriched_data['stock_theorique'] = 120.0

        result = run_enriched_notebook_analysis(
            df=sample_enriched_data,
            product_name='Poulet Frais',
            periods=14,
        )

        assert len(result['predictions_futures']) == 14
        assert 'quantite_a_commander' in result['recommendation']
        assert result['recommendation']['horizon_commande_jours'] == 14

    def test_builds_daily_arrivals_for_one_day_dlc(self):
        produit_df = pd.DataFrame(
            {
                'date': pd.to_datetime(['2024-12-30']),
                'stock_theorique': [0.0],
                'date_expiration': pd.to_datetime(['2024-12-31']),
            }
        )
        predictions_futures = pd.DataFrame(
            {
                'ds': pd.date_range('2024-12-31', periods=3, freq='D'),
                'yhat': [5.0, 8.0, 7.0],
                'yhat_lower': [4.0, 7.0, 6.0],
                'yhat_upper': [6.0, 9.0, 8.0],
            }
        )

        arrival_plan = build_dlc_arrival_schedule(
            produit_df=produit_df,
            predictions_futures=predictions_futures,
            product_name='Produit Test',
            dlc_days=1,
            current_usable_stock=0.0,
            current_stock_remaining_days=1,
        )

        assert len(arrival_plan) == 3
        assert arrival_plan['quantity_to_order'].tolist() == [5.0, 8.0, 7.0]
        assert arrival_plan['window_days'].tolist() == [1, 1, 1]

    def test_groups_arrivals_by_dlc_window_and_uses_current_stock(self):
        produit_df = pd.DataFrame(
            {
                'date': pd.to_datetime(['2024-12-30']),
                'stock_theorique': [0.0],
                'date_expiration': pd.to_datetime(['2025-01-02']),
            }
        )
        predictions_futures = pd.DataFrame(
            {
                'ds': pd.date_range('2024-12-31', periods=4, freq='D'),
                'yhat': [5.0, 6.0, 7.0, 8.0],
                'yhat_lower': [4.0, 5.0, 6.0, 7.0],
                'yhat_upper': [6.0, 7.0, 8.0, 9.0],
            }
        )

        arrival_plan = build_dlc_arrival_schedule(
            produit_df=produit_df,
            predictions_futures=predictions_futures,
            product_name='Produit Test',
            dlc_days=3,
            current_usable_stock=4.0,
            current_stock_remaining_days=3,
        )

        assert len(arrival_plan) == 2
        assert arrival_plan['quantity_to_order'].round(2).tolist() == [14.0, 8.0]
        assert arrival_plan['current_stock_used'].round(2).tolist() == [4.0, 0.0]
        assert arrival_plan['window_days'].tolist() == [3, 1]