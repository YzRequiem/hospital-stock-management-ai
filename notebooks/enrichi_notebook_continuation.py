"""
Continuation du notebook enrichi - Cellules à ajouter
Ce fichier contient le code pour les étapes 6-12 du notebook
"""

# ============================================================================
# ÉTAPE 6 : CONFIGURATION DES HOLIDAYS
# ============================================================================

CELL_MARKDOWN_STEP6 = """
# 📅 Étape 6 : Configuration des Holidays (Jours Fériés)

Prophet permet de modéliser des événements spéciaux via le système de "holidays".
Nous allons créer 3 types de holidays :
1. **Jours fériés français**
2. **Vacances scolaires**
3. **Périodes COVID** (événement majeur)
"""

CELL_CODE_STEP6_1 = """
# Créer le DataFrame des jours fériés
print("📅 CONFIGURATION DES HOLIDAYS")
print("="*70)

# 1. Jours fériés français
holidays_jf = daily[daily['jour_ferie'] == 1][['date']].drop_duplicates()
holidays_jf.columns = ['ds']
holidays_jf['holiday'] = 'jour_ferie'
holidays_jf['lower_window'] = 0
holidays_jf['upper_window'] = 0

print(f"\\n✅ Jours fériés : {len(holidays_jf)} jours")

# 2. Vacances scolaires
holidays_vac = daily[daily['vacances_scolaires'] == 1][['date']].drop_duplicates()
holidays_vac.columns = ['ds']
holidays_vac['holiday'] = 'vacances_scolaires'
holidays_vac['lower_window'] = 0
holidays_vac['upper_window'] = 0

print(f"✅ Vacances scolaires : {len(holidays_vac)} jours")

# 3. Périodes COVID (événement majeur)
holidays_covid = daily[daily['covid_impact'] == 1][['date']].drop_duplicates()
holidays_covid.columns = ['ds']
holidays_covid['holiday'] = 'covid_19'
holidays_covid['lower_window'] = 0
holidays_covid['upper_window'] = 0

print(f"✅ Périodes COVID : {len(holidays_covid)} jours")

# Combiner tous les holidays
holidays = pd.concat([holidays_jf, holidays_vac, holidays_covid])

print(f"\\n📊 Total holidays configurés : {len(holidays)} jours")
print(f"\\nTypes de holidays :")
print(holidays['holiday'].value_counts())
"""

CELL_CODE_STEP6_2 = """
# Aperçu des holidays
print("\\n👀 Aperçu des 10 premiers holidays :")
holidays.head(10)
"""

# ============================================================================
# ÉTAPE 7 : CONFIGURATION DES CHANGEPOINTS
# ============================================================================

CELL_MARKDOWN_STEP7 = """
# 🔄 Étape 7 : Configuration des Changepoints

Les changepoints sont des moments où la tendance du modèle peut changer.
Prophet peut les détecter automatiquement, mais nous pouvons aussi les spécifier manuellement.

### Changepoints connus dans notre dataset :
1. **15/03/2020** - COVID-19 Vague 1 (+50%)
2. **01/11/2020** - COVID-19 Vague 2 (+30%)
3. **01/05/2021** - Déconfinement (-10%)
4. **01/01/2022** - Nouvelle Direction (+10%)
5. **01/09/2023** - Extension Hôpital (+15%)
"""

CELL_CODE_STEP7 = """
# Définir les changepoints manuellement
print("🔄 CONFIGURATION DES CHANGEPOINTS")
print("="*70)

# Dates des événements majeurs
changepoints_manuels = [
    '2020-03-15',  # COVID-19 Vague 1
    '2020-11-01',  # COVID-19 Vague 2
    '2021-05-01',  # Déconfinement
    '2022-01-01',  # Nouvelle Direction
    '2023-09-01'   # Extension Hôpital
]

print(f"\\n✅ {len(changepoints_manuels)} changepoints manuels définis :")
for i, cp in enumerate(changepoints_manuels, 1):
    print(f"   {i}. {cp}")

print(f"\\n💡 Note : Prophet peut aussi détecter automatiquement d'autres changepoints")
print(f"   → Paramètre : changepoint_prior_scale (flexibilité)")
print(f"   → Valeur par défaut : 0.05")
print(f"   → Valeur recommandée pour ce dataset : 0.5 (plus flexible)")
"""

# ============================================================================
# ÉTAPE 8 : SPLIT TRAIN/TEST
# ============================================================================

CELL_MARKDOWN_STEP8 = """
# 📊 Étape 8 : Split Train/Test

Nous allons diviser les données en :
- **Train** : 80% des données (environ 4 ans)
- **Test** : 20% des données (environ 1 an)

Cela nous permettra d'évaluer la performance du modèle.
"""

CELL_CODE_STEP8 = """
# Split train/test
print("📊 SPLIT TRAIN/TEST")
print("="*70)

# Garder 20% pour test (environ 1 an sur 5 ans)
split_date = prophet_df['ds'].max() - pd.Timedelta(days=365)
train = prophet_df[prophet_df['ds'] <= split_date].copy()
test = prophet_df[prophet_df['ds'] > split_date].copy()

print(f"\\n✅ Données divisées :")
print(f"   Train : {len(train)} jours ({train['ds'].min().date()} → {train['ds'].max().date()})")
print(f"   Test  : {len(test)} jours ({test['ds'].min().date()} → {test['ds'].max().date()})")
print(f"\\n   Ratio : {len(train)/len(prophet_df)*100:.1f}% train / {len(test)/len(prophet_df)*100:.1f}% test")
"""

# ============================================================================
# ÉTAPE 9 : MODÈLE PROPHET AVEC REGRESSORS
# ============================================================================

CELL_MARKDOWN_STEP9 = """
# 🤖 Étape 9 : Modèle Prophet avec Regressors

Nous allons configurer un modèle Prophet avancé avec :
1. **Holidays** (jours fériés, vacances, COVID)
2. **Changepoints** manuels
3. **Seasonality** optimisée (yearly + weekly)
4. **Regressors** externes (temperature, occupation, patients, grippe)
"""

CELL_CODE_STEP9_1 = """
# Configuration du modèle Prophet
print("🤖 CONFIGURATION DU MODÈLE PROPHET")
print("="*70)

# Créer le modèle avec configuration optimale
model = Prophet(
    # ===== HOLIDAYS =====
    holidays=holidays,
    holidays_prior_scale=10.0,        # Importance des holidays (default: 10)

    # ===== SEASONALITY =====
    yearly_seasonality=20,            # Ordre Fourier (default: 10, max: 20)
    weekly_seasonality=5,             # Ordre Fourier (default: 3)
    daily_seasonality=False,          # Pas nécessaire pour données quotidiennes
    seasonality_mode='multiplicative', # 'multiplicative' ou 'additive'
    seasonality_prior_scale=10.0,     # Flexibilité saisonnalité (default: 10)

    # ===== CHANGEPOINTS =====
    changepoints=changepoints_manuels, # Changepoints manuels
    changepoint_prior_scale=0.5,      # Flexibilité (default: 0.05)
    changepoint_range=0.9,            # 90% des données (default: 0.8)

    # ===== AUTRES =====
    interval_width=0.85,              # Intervalle de confiance 85%
    growth='linear',                  # Croissance linéaire
    mcmc_samples=0                    # Bayésien si > 0
)

print("\\n✅ Modèle créé avec :")
print(f"   - Holidays : {len(holidays)} événements")
print(f"   - Changepoints : {len(changepoints_manuels)} manuels")
print(f"   - Seasonality : yearly (20) + weekly (5)")
"""

CELL_CODE_STEP9_2 = """
# Ajouter les régresseurs
print("\\n🔧 AJOUT DES RÉGRESSEURS")
print("="*70)

# 1. Température (effet sur produits frais)
model.add_regressor(
    'temperature',
    prior_scale=0.5,      # Importance du régresseur
    standardize=True,     # Normaliser automatiquement
    mode='additive'       # Effet additif
)
print("✅ Régresseur 'temperature' ajouté")

# 2. Taux d'occupation (corrélation forte avec consommation)
model.add_regressor(
    'taux_occupation',
    prior_scale=1.0,      # Plus important
    standardize=True,
    mode='additive'
)
print("✅ Régresseur 'taux_occupation' ajouté")

# 3. Nombre de patients (alternative au taux occupation)
model.add_regressor(
    'nb_patients',
    prior_scale=0.5,
    standardize=True,
    mode='additive'
)
print("✅ Régresseur 'nb_patients' ajouté")

# 4. Épidémie de grippe (effet fort sur certains produits)
model.add_regressor(
    'epidemie_grippe',
    prior_scale=0.5,
    standardize=False,    # Déjà binaire (0 ou 1)
    mode='additive'
)
print("✅ Régresseur 'epidemie_grippe' ajouté")

print("\\n📊 4 régresseurs configurés")
"""

CELL_CODE_STEP9_3 = """
# Entraîner le modèle
print("\\n⏳ ENTRAÎNEMENT DU MODÈLE...")
print("="*70)
print("⏳ Cela peut prendre 1-2 minutes...")

model.fit(train)

print("\\n✅ Modèle entraîné avec succès !")
"""

# ============================================================================
# ÉTAPE 10 : ÉVALUATION SUR LE TEST
# ============================================================================

CELL_MARKDOWN_STEP10 = """
# 📊 Étape 10 : Évaluation sur le Test

Évaluons la performance du modèle sur les données de test.
"""

CELL_CODE_STEP10 = """
# Prédire sur test
print("📊 ÉVALUATION SUR LE TEST")
print("="*70)

predictions_test = model.predict(test)

# Calculer les métriques
y_true = test['y'].values
y_pred = predictions_test['yhat'].values

mae = np.mean(np.abs(y_true - y_pred))

# MAPE calculé uniquement sur les jours avec consommation > 0
mask_nonzero = y_true > 0
if mask_nonzero.sum() > 0:
    mape = np.mean(np.abs((y_true[mask_nonzero] - y_pred[mask_nonzero]) / y_true[mask_nonzero])) * 100
else:
    mape = 0.0

rmse = np.sqrt(np.mean((y_true - y_pred)**2))

print(f"\\n📊 MÉTRIQUES DE PERFORMANCE")
print("="*70)
print(f"MAE  (Erreur Absolue Moyenne)    : {mae:.2f} kg")
print(f"MAPE (Erreur Relative Moyenne)   : {mape:.2f}%")
print(f"RMSE (Erreur Quadratique Moyenne): {rmse:.2f} kg")
print("="*70)

if mape < 15:
    print("\\n✅ Excellente précision ! (MAPE < 15%)")
elif mape < 25:
    print("\\n✅ Bonne précision (MAPE < 25%)")
else:
    print("\\n⚠️  Précision moyenne (MAPE > 25%)")

print(f"\\n💡 Le modèle se trompe en moyenne de ±{mae:.2f} kg (soit {mape:.1f}%)")
"""

# À SUIVRE...
# Les étapes 11-12 seront dans le prochain fichier
print("✅ Cellules 6-10 prêtes à être ajoutées au notebook")
