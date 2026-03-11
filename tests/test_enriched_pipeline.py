"""Tests for the notebook-aligned enriched pipeline."""

from pathlib import Path

import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.enriched_pipeline import run_enriched_notebook_analysis


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
        assert result['recommendation']['horizon_commande_jours'] == 7