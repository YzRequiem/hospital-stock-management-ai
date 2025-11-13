#!/usr/bin/env python3
"""
Analyse Complète avec Dataset Enrichi - Clinique du Mont Vert
Utilisation avancée de Prophet avec regressors externes

Usage:
    python analyse_enrichie_complete.py

Author: Claude Code
Dataset: dataset_stock_hopital_ENRICHI.csv (v3.0)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import json
from pathlib import Path

warnings.filterwarnings('ignore')

# Configuration
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:.2f}'.format)
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (16, 8)

# ============================================================================
# ÉTAPE 1 : IMPORTS ET CONFIGURATION
# ============================================================================

print("="*70)
print("🏥 ANALYSE AVANCÉE AVEC DATASET ENRICHI")
print("="*70)

try:
    from prophet import Prophet
    from prophet.plot import add_changepoints_to_plot
    from prophet.utilities import regressor_coefficients
    print("✅ Prophet importé avec succès")
    PROPHET_AVAILABLE = True
except ImportError:
    print("❌ Prophet non disponible. Installation : pip install prophet")
    exit(1)

# Results Manager (optionnel)
try:
    from results_manager import ResultsManager
    results_mgr = ResultsManager()
    results_mgr.create_run_directory()
    print(f"✅ Results Manager : {results_mgr.get_run_path()}")
    USE_RESULTS_MANAGER = True
except:
    print("⚠️  Results Manager non disponible")
    USE_RESULTS_MANAGER = False

# ============================================================================
# ÉTAPE 2 : CHARGEMENT DU DATASET ENRICHI
# ============================================================================

print("\\n" + "="*70)
print("📂 CHARGEMENT DU DATASET ENRICHI")
print("="*70)

FICHIER_CSV = "../data/dataset_stock_hopital_ENRICHI.csv"
PRODUIT_ANALYSE = "Poulet frais"  # ← Changez ici

if not Path(FICHIER_CSV).exists():
    print(f"❌ Fichier introuvable : {FICHIER_CSV}")
    exit(1)

df = pd.read_csv(FICHIER_CSV)
df['date'] = pd.to_datetime(df['date'])

print(f"✅ Dataset : {len(df):,} lignes × {len(df.columns)} colonnes")
print(f"📅 Période : {df['date'].min().date()} → {df['date'].max().date()}")

# ============================================================================
# ÉTAPE 3 : FILTRAGE ET AGRÉGATION
# ============================================================================

print("\\n" + "="*70)
print(f"🔍 ANALYSE DU PRODUIT : {PRODUIT_ANALYSE}")
print("="*70)

produit_df = df[
    (df['nom_produit'] == PRODUIT_ANALYSE) &
    (df['type_sortie'] == 'CONSOMMATION')
].copy()

print(f"✅ {len(produit_df):,} sorties trouvées")

# Agrégation quotidienne
daily = produit_df.groupby('date').agg({
    'quantite': 'sum',
    'temperature': 'mean',
    'taux_occupation': 'mean',
    'nb_patients': 'mean',
    'epidemie_grippe': 'max',
    'vacances_scolaires': 'max',
    'jour_ferie': 'max',
    'covid_impact': 'max'
}).reset_index()

# Compléter les dates manquantes
date_range = pd.date_range(
    start=daily['date'].min(),
    end=daily['date'].max(),
    freq='D'
)
full_dates = pd.DataFrame({'date': date_range})
daily = full_dates.merge(daily, on='date', how='left')

# Remplir les NaN
daily['quantite'].fillna(0, inplace=True)
for col in ['temperature', 'taux_occupation', 'nb_patients']:
    daily[col].fillna(daily[col].mean(), inplace=True)
for col in ['epidemie_grippe', 'vacances_scolaires', 'jour_ferie', 'covid_impact']:
    daily[col].fillna(0, inplace=True)

print(f"✅ {len(daily)} jours préparés")

# Renommer pour Prophet
prophet_df = daily.rename(columns={'date': 'ds', 'quantite': 'y'})

# ============================================================================
# ÉTAPE 4 : CONFIGURATION DES HOLIDAYS
# ============================================================================

print("\\n" + "="*70)
print("📅 CONFIGURATION DES HOLIDAYS")
print("="*70)

# Jours fériés
holidays_jf = daily[daily['jour_ferie'] == 1][['date']].drop_duplicates()
holidays_jf.columns = ['ds']
holidays_jf['holiday'] = 'jour_ferie'
holidays_jf['lower_window'] = 0
holidays_jf['upper_window'] = 0

# Vacances scolaires
holidays_vac = daily[daily['vacances_scolaires'] == 1][['date']].drop_duplicates()
holidays_vac.columns = ['ds']
holidays_vac['holiday'] = 'vacances_scolaires'
holidays_vac['lower_window'] = 0
holidays_vac['upper_window'] = 0

# COVID
holidays_covid = daily[daily['covid_impact'] == 1][['date']].drop_duplicates()
holidays_covid.columns = ['ds']
holidays_covid['holiday'] = 'covid_19'
holidays_covid['lower_window'] = 0
holidays_covid['upper_window'] = 0

holidays = pd.concat([holidays_jf, holidays_vac, holidays_covid])

print(f"✅ {len(holidays)} holidays configurés")
print(f"   - Jours fériés : {len(holidays_jf)}")
print(f"   - Vacances : {len(holidays_vac)}")
print(f"   - COVID : {len(holidays_covid)}")

# ============================================================================
# ÉTAPE 5 : CHANGEPOINTS
# ============================================================================

changepoints_manuels = [
    '2020-03-15',  # COVID-19 Vague 1
    '2020-11-01',  # COVID-19 Vague 2
    '2021-05-01',  # Déconfinement
    '2022-01-01',  # Nouvelle Direction
    '2023-09-01'   # Extension Hôpital
]

print(f"✅ {len(changepoints_manuels)} changepoints manuels")

# ============================================================================
# ÉTAPE 6 : SPLIT TRAIN/TEST
# ============================================================================

print("\\n" + "="*70)
print("📊 SPLIT TRAIN/TEST")
print("="*70)

split_date = prophet_df['ds'].max() - pd.Timedelta(days=365)
train = prophet_df[prophet_df['ds'] <= split_date].copy()
test = prophet_df[prophet_df['ds'] > split_date].copy()

print(f"✅ Train : {len(train)} jours ({train['ds'].min().date()} → {train['ds'].max().date()})")
print(f"✅ Test  : {len(test)} jours ({test['ds'].min().date()} → {test['ds'].max().date()})")

# ============================================================================
# ÉTAPE 7 : MODÈLE PROPHET
# ============================================================================

print("\\n" + "="*70)
print("🤖 CONFIGURATION ET ENTRAÎNEMENT DU MODÈLE")
print("="*70)

model = Prophet(
    holidays=holidays,
    holidays_prior_scale=10.0,
    yearly_seasonality=20,
    weekly_seasonality=5,
    daily_seasonality=False,
    seasonality_mode='multiplicative',
    seasonality_prior_scale=10.0,
    changepoints=changepoints_manuels,
    changepoint_prior_scale=0.5,
    changepoint_range=0.9,
    interval_width=0.85,
    growth='linear'
)

# Ajouter les régresseurs
model.add_regressor('temperature', prior_scale=0.5, standardize=True)
model.add_regressor('taux_occupation', prior_scale=1.0, standardize=True)
model.add_regressor('nb_patients', prior_scale=0.5, standardize=True)
model.add_regressor('epidemie_grippe', prior_scale=0.5, standardize=False)

print("✅ Modèle configuré avec 4 régresseurs")
print("⏳ Entraînement en cours...")

model.fit(train)

print("✅ Modèle entraîné !")

# ============================================================================
# ÉTAPE 8 : ÉVALUATION
# ============================================================================

print("\\n" + "="*70)
print("📊 ÉVALUATION SUR LE TEST")
print("="*70)

predictions_test = model.predict(test)

y_true = test['y'].values
y_pred = predictions_test['yhat'].values

mae = np.mean(np.abs(y_true - y_pred))
mape = np.mean(np.abs((y_true - y_pred) / (y_true + 0.01))) * 100
rmse = np.sqrt(np.mean((y_true - y_pred)**2))

print(f"\\nMAE  : {mae:.2f} kg")
print(f"MAPE : {mape:.2f}%")
print(f"RMSE : {rmse:.2f} kg")

if mape < 15:
    print("\\n✅ Excellente précision ! (MAPE < 15%)")
elif mape < 25:
    print("\\n✅ Bonne précision (MAPE < 25%)")
else:
    print("\\n⚠️  Précision moyenne (MAPE > 25%)")

# ============================================================================
# ÉTAPE 9 : ANALYSE DES COEFFICIENTS
# ============================================================================

print("\\n" + "="*70)
print("📊 COEFFICIENTS DES RÉGRESSEURS")
print("="*70)

try:
    coeffs = regressor_coefficients(model)
    print(coeffs)

    print("\\n💡 Interprétation :")
    for idx, row in coeffs.iterrows():
        regressor = row['regressor']
        coeff = row['coef']

        if coeff > 0:
            print(f"   📈 {regressor:20s} : +{coeff:.4f} (effet positif)")
        else:
            print(f"   📉 {regressor:20s} : {coeff:.4f} (effet négatif)")
except Exception as e:
    print(f"⚠️  Impossible d'extraire les coefficients : {e}")

# ============================================================================
# ÉTAPE 10 : PRÉDICTIONS FUTURES
# ============================================================================

print("\\n" + "="*70)
print("🔮 PRÉDICTIONS FUTURES (28 JOURS)")
print("="*70)

# Réentraîner sur toutes les données
model_final = Prophet(
    holidays=holidays,
    holidays_prior_scale=10.0,
    yearly_seasonality=20,
    weekly_seasonality=5,
    seasonality_mode='multiplicative',
    changepoints=changepoints_manuels,
    changepoint_prior_scale=0.5,
    interval_width=0.85
)

model_final.add_regressor('temperature', prior_scale=0.5, standardize=True)
model_final.add_regressor('taux_occupation', prior_scale=1.0, standardize=True)
model_final.add_regressor('nb_patients', prior_scale=0.5, standardize=True)
model_final.add_regressor('epidemie_grippe', prior_scale=0.5, standardize=False)

model_final.fit(prophet_df)

# Créer les dates futures
future = model_final.make_future_dataframe(periods=28)

# Ajouter les valeurs des régresseurs (moyenne pour le futur)
future = future.merge(
    prophet_df[['ds', 'temperature', 'taux_occupation', 'nb_patients', 'epidemie_grippe']],
    on='ds',
    how='left'
)

for col in ['temperature', 'taux_occupation', 'nb_patients']:
    future[col].fillna(prophet_df[col].mean(), inplace=True)

# Épidémie de grippe : activer si hiver (jan-mars)
future['epidemie_grippe'].fillna(
    future['ds'].dt.month.isin([1, 2, 3]).astype(int),
    inplace=True
)

# Prédire
forecast = model_final.predict(future)
predictions_futures = forecast[forecast['ds'] > prophet_df['ds'].max()]

print(f"✅ {len(predictions_futures)} jours de prédictions")
print(f"   Total prévu : {predictions_futures['yhat'].sum():.2f} kg")
print(f"   Moyenne/jour : {predictions_futures['yhat'].mean():.2f} kg")

# ============================================================================
# ÉTAPE 11 : VISUALISATIONS
# ============================================================================

print("\\n" + "="*70)
print("📈 GÉNÉRATION DES VISUALISATIONS")
print("="*70)

# Graphique principal
fig1 = model_final.plot(forecast)
plt.title(f'Prédictions Prophet - {PRODUIT_ANALYSE}', fontsize=14, fontweight='bold')
filename1 = f'predictions_{PRODUIT_ANALYSE.replace(" ", "_")}_enrichi.png'
plt.savefig(filename1, dpi=300, bbox_inches='tight')
print(f"✅ {filename1}")
plt.close()

# Composants
fig2 = model_final.plot_components(forecast)
filename2 = f'components_{PRODUIT_ANALYSE.replace(" ", "_")}_enrichi.png'
plt.savefig(filename2, dpi=300, bbox_inches='tight')
print(f"✅ {filename2}")
plt.close()

# Changepoints
fig3 = model_final.plot(forecast)
add_changepoints_to_plot(fig3.gca(), model_final, forecast)
plt.title(f'Changepoints - {PRODUIT_ANALYSE}', fontsize=14, fontweight='bold')
filename3 = f'changepoints_{PRODUIT_ANALYSE.replace(" ", "_")}_enrichi.png'
plt.savefig(filename3, dpi=300, bbox_inches='tight')
print(f"✅ {filename3}")
plt.close()

# ============================================================================
# ÉTAPE 12 : EXPORT DES RÉSULTATS
# ============================================================================

print("\\n" + "="*70)
print("💾 EXPORT DES RÉSULTATS")
print("="*70)

# Export CSV
export_df = predictions_futures[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
export_df.columns = ['date', 'quantite_prevue', 'quantite_min', 'quantite_max']
export_df['date'] = export_df['date'].dt.date
export_df['produit'] = PRODUIT_ANALYSE
export_df['confiance'] = '85%'

filename_csv = f'predictions_{PRODUIT_ANALYSE.replace(" ", "_")}_enrichi_28j.csv'
export_df.to_csv(filename_csv, index=False)
print(f"✅ {filename_csv}")

# Export JSON
summary = {
    "produit": PRODUIT_ANALYSE,
    "date_analyse": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    "performance_modele": {
        "MAE": round(mae, 2),
        "MAPE": round(mape, 2),
        "RMSE": round(rmse, 2),
        "methode": "Prophet avec regressors enrichis"
    },
    "regresseurs_utilises": [
        "temperature", "taux_occupation", "nb_patients", "epidemie_grippe"
    ],
    "holidays_utilises": [
        "jour_ferie", "vacances_scolaires", "covid_19"
    ],
    "changepoints": changepoints_manuels,
    "predictions": {
        "horizon": "28 jours",
        "total_prevu": round(predictions_futures['yhat'].sum(), 2),
        "moyenne_jour": round(predictions_futures['yhat'].mean(), 2)
    }
}

filename_json = f'summary_{PRODUIT_ANALYSE.replace(" ", "_")}_enrichi.json'
with open(filename_json, 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(f"✅ {filename_json}")

# Sauvegarder avec Results Manager si disponible
if USE_RESULTS_MANAGER:
    for f in [filename1, filename2, filename3]:
        results_mgr.save_graph(f)
    results_mgr.save_data(filename_csv)
    results_mgr.save_data(filename_json)
    results_mgr.create_summary_file(PRODUIT_ANALYSE, summary)
    print(f"\\n✅ Résultats sauvegardés dans : {results_mgr.get_run_path()}")

print("\\n" + "="*70)
print("🎉 ANALYSE TERMINÉE AVEC SUCCÈS !")
print("="*70)
