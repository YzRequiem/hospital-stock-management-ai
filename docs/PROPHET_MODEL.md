# 🔮 Guide du Modèle Prophet

Documentation complète sur l'utilisation de Prophet dans le projet de gestion de stock hospitalier.

---

## 📚 Table des matières

1. [Qu'est-ce que Prophet ?](#quest-ce-que-prophet-)
2. [Pourquoi Prophet pour ce projet ?](#pourquoi-prophet-pour-ce-projet-)
3. [Fonctionnement de Prophet](#fonctionnement-de-prophet)
4. [Notre implémentation](#notre-implémentation)
5. [Configuration des paramètres](#configuration-des-paramètres)
6. [Les régresseurs externes](#les-régresseurs-externes)
7. [Métriques d'évaluation](#métriques-dévaluation)
8. [Exemples d'utilisation](#exemples-dutilisation)

---

## Qu'est-ce que Prophet ?

**Prophet** est un outil de prévision de séries temporelles développé par **Meta (Facebook)** et publié en open source en 2017.

### Caractéristiques principales

| Aspect          | Description                                                      |
| --------------- | ---------------------------------------------------------------- |
| **Type**        | Modèle de décomposition additive/multiplicative                  |
| **Créateur**    | Meta (Facebook) Core Data Science                                |
| **Langage**     | Python et R                                                      |
| **Licence**     | MIT (open source)                                                |
| **Publication** | [Forecasting at Scale](https://peerj.com/preprints/3190/) (2017) |

### Équation du modèle

Prophet décompose une série temporelle selon l'équation :

$$y(t) = g(t) + s(t) + h(t) + \epsilon_t$$

Où :

- **$g(t)$** : Tendance (croissance linéaire ou logistique)
- **$s(t)$** : Saisonnalité (patterns répétitifs)
- **$h(t)$** : Effet des jours fériés/événements
- **$\epsilon_t$** : Terme d'erreur (bruit)

---

## Pourquoi Prophet pour ce projet ?

### ✅ Avantages pour la prédiction de stock hospitalier

| Avantage                                       | Application dans notre projet                                       |
| ---------------------------------------------- | ------------------------------------------------------------------- |
| **Gestion automatique des données manquantes** | Les jours sans consommation sont gérés automatiquement              |
| **Saisonnalité multiple**                      | Capture des patterns hebdomadaires (weekend) et annuels (été/hiver) |
| **Régresseurs externes**                       | Intégration de la température, occupation, épidémies...             |
| **Robuste aux outliers**                       | Résiste aux pics de consommation exceptionnels                      |
| **Intervalles de confiance**                   | Quantification de l'incertitude pour la planification               |
| **Rapide à entraîner**                         | Quelques secondes même sur 5 ans de données                         |
| **Pas d'expertise ML requise**                 | Paramètres intuitifs et bien documentés                             |

### ❌ Pourquoi pas d'autres modèles ?

| Modèle             | Raison du non-choix                                         |
| ------------------ | ----------------------------------------------------------- |
| **ARIMA/SARIMA**   | Complexe à paramétrer, sensible aux données manquantes      |
| **LSTM/RNN**       | Nécessite beaucoup de données et de tuning                  |
| **XGBoost**        | Moins adapté aux séries temporelles avec saisonnalité forte |
| **Moyenne mobile** | Trop simpliste, pas de saisonnalité                         |

### 🎯 Cas d'usage idéal

Prophet est conçu pour les séries temporelles qui ont :

- ✅ **Une forte saisonnalité** → Pattern hebdomadaire hôpital
- ✅ **Plusieurs saisons d'historique** → 5 ans de données (2020-2024)
- ✅ **Des données manquantes** → Jours sans consommation
- ✅ **Des changements de tendance** → Impact COVID, nouvelles pratiques
- ✅ **Des événements spéciaux** → Jours fériés, épidémies

---

## Fonctionnement de Prophet

### 1. La tendance $g(t)$

Prophet modélise la tendance avec une fonction linéaire par morceaux :

```
g(t) = (k + a(t)ᵀδ) * t + (m + a(t)ᵀγ)
```

- **k** : Taux de croissance de base
- **δ** : Ajustements du taux aux points de changement
- **m** : Offset
- **γ** : Ajustements pour maintenir la continuité

**Changepoints** : Prophet détecte automatiquement les moments où la tendance change (ex: impact COVID).

### 2. La saisonnalité $s(t)$

Prophet utilise des **séries de Fourier** pour modéliser la saisonnalité :

$$s(t) = \sum_{n=1}^{N} \left( a_n \cos\left(\frac{2\pi nt}{P}\right) + b_n \sin\left(\frac{2\pi nt}{P}\right) \right)$$

Dans notre projet :

| Saisonnalité     | Période P    | Ordre N | Pattern capté       |
| ---------------- | ------------ | ------- | ------------------- |
| **Hebdomadaire** | 7 jours      | 3       | Weekend vs semaine  |
| **Annuelle**     | 365.25 jours | 10      | Été vs hiver, fêtes |

### 3. Les événements $h(t)$

Les jours fériés et événements spéciaux sont modélisés comme des indicateurs :

```python
holidays = pd.DataFrame({
    'holiday': 'jour_ferie',
    'ds': ['2024-01-01', '2024-05-01', '2024-07-14', ...],
    'lower_window': 0,  # Jours avant
    'upper_window': 1   # Jours après
})
```

---

## Notre implémentation

### Architecture du code

```
src/
├── model.py          # Fonctions Prophet
├── data_loader.py    # Préparation des données
└── metrics.py        # Évaluation des performances

config/
├── model_params.yaml # Paramètres Prophet
└── settings.py       # Configuration générale
```

### Fonctions principales (`src/model.py`)

```python
# Créer un modèle configuré
model = create_prophet_model(
    daily_seasonality=False,
    weekly_seasonality=True,
    yearly_seasonality=True,
    seasonality_mode='additive',
    changepoint_prior_scale=0.05,
    regressors=['temperature', 'nb_patients']
)

# Entraîner le modèle
model = train_prophet_model(train_df, **config)

# Prédire sur données existantes
predictions = predict(model, test_df)

# Prédire le futur
future_predictions = predict_future(model, periods=30)
```

### Pipeline complet

```
┌─────────────────┐
│  Dataset CSV    │
│  (85k lignes)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  load_dataset() │
│  Encodage auto  │
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│  prepare_prophet_data() │
│  - Filtre produit       │
│  - Agrège par jour      │
│  - Renomme ds/y         │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  train_test_split()     │
│  80% train / 20% test   │
│  (split chronologique)  │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  train_prophet_model()  │
│  - Configure Prophet    │
│  - Ajoute régresseurs   │
│  - Entraîne (~1-2 min)  │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  predict() / predict_   │
│  future()               │
│  - Génère prédictions   │
│  - Intervalles confiance│
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  calculate_metrics()    │
│  MAE, MAPE, RMSE, R²    │
└─────────────────────────┘
```

---

## Configuration des paramètres

### Paramètres principaux (`config/model_params.yaml`)

```yaml
default:
  # Saisonnalité
  daily_seasonality: false # Pas de pattern intra-journalier
  weekly_seasonality: true # Pattern fort semaine/weekend
  yearly_seasonality: true # Variations saisonnières
  seasonality_mode: "additive"

  # Flexibilité
  changepoint_prior_scale: 0.05 # Équilibre biais/variance

  # Intervalles de confiance
  interval_width: 0.85 # 85% de confiance
```

### Guide des paramètres

| Paramètre                 | Valeur         | Effet                              |
| ------------------------- | -------------- | ---------------------------------- |
| `changepoint_prior_scale` | 0.001          | Tendance très lisse                |
|                           | **0.05**       | Équilibré (défaut)                 |
|                           | 0.5            | Très flexible (risque overfitting) |
| `seasonality_mode`        | **additive**   | Variations constantes              |
|                           | multiplicative | Variations proportionnelles        |
| `interval_width`          | 0.80           | Intervalles étroits                |
|                           | **0.85**       | Standard                           |
|                           | 0.95           | Intervalles larges                 |

### Configurations par type de produit

```yaml
# Produits à DLC courte (viande, poisson)
short_dlc:
  changepoint_prior_scale: 0.1 # Plus réactif
  interval_width: 0.90 # Plus d'incertitude

# Produits à DLC longue (conserves)
long_dlc:
  changepoint_prior_scale: 0.01 # Plus stable
  interval_width: 0.80 # Moins d'incertitude
```

---

## Les régresseurs externes

### Régresseurs disponibles dans le dataset enrichi

Notre dataset enrichi (85k lignes) contient **7 régresseurs externes** qui améliorent significativement les prédictions :

| Régresseur           | Type    | Description                  | Impact attendu                          |
| -------------------- | ------- | ---------------------------- | --------------------------------------- |
| `temperature`        | float   | Température journalière (°C) | Soupes en hiver, salades en été         |
| `taux_occupation`    | float   | % d'occupation des lits      | Impact direct proportionnel             |
| `nb_patients`        | int     | Nombre de patients/jour      | Plus de patients = plus de consommation |
| `epidemie_grippe`    | binaire | Période d'épidémie           | Augmentation consommation               |
| `vacances_scolaires` | binaire | Période vacances             | Réduction personnel                     |
| `jour_ferie`         | binaire | Jour férié                   | Menu simplifié                          |
| `covid_impact`       | binaire | Période COVID                | Perturbation majeure                    |

### Comment les utiliser

```python
from src.model import train_prophet_model

# Définir les régresseurs à utiliser
regressors = [
    'temperature',
    'taux_occupation',
    'nb_patients',
    'epidemie_grippe'
]

# Entraîner avec régresseurs
model = train_prophet_model(
    train_df,
    regressors=regressors,
    weekly_seasonality=True,
    yearly_seasonality=True
)
```

### Amélioration des performances

| Configuration            | MAPE typique |
| ------------------------ | ------------ |
| Prophet de base          | 25-35%       |
| + Saisonnalité optimisée | 20-28%       |
| + Régresseurs externes   | **15-22%**   |
| + Holidays configurés    | **12-18%**   |

---

## Métriques d'évaluation

### Métriques calculées (`src/metrics.py`)

| Métrique | Formule                                                       | Interprétation               |
| -------- | ------------------------------------------------------------- | ---------------------------- |
| **MAE**  | $\frac{1}{n}\sum\|y_i - \hat{y}_i\|$                          | Erreur moyenne en kg         |
| **MAPE** | $\frac{100}{n}\sum\left\|\frac{y_i - \hat{y}_i}{y_i}\right\|$ | Erreur relative en %         |
| **RMSE** | $\sqrt{\frac{1}{n}\sum(y_i - \hat{y}_i)^2}$                   | Pénalise les grosses erreurs |
| **R²**   | $1 - \frac{SS_{res}}{SS_{tot}}$                               | Variance expliquée           |

### Interprétation du MAPE

```
MAPE < 10%  → ✅ Excellent (prédiction très fiable)
MAPE < 15%  → ✅ Très bon (recommandation automatique possible)
MAPE < 25%  → ✅ Bon (aide à la décision)
MAPE < 50%  → ⚠️ Acceptable (utiliser avec précaution)
MAPE > 50%  → ❌ Insuffisant (revoir le modèle)
```

### Notre implémentation du MAPE

```python
def calculate_mape(y_true, y_pred, exclude_zeros=True):
    """
    Calcule MAPE en excluant les jours à zéro.

    Pourquoi exclude_zeros=True ?
    - Les jours sans consommation (y=0) donnent MAPE = infini
    - Ce sont souvent des weekends/fériés (pattern, pas erreur)
    - On veut mesurer la précision sur les jours actifs
    """
    if exclude_zeros:
        mask = y_true > 0
        y_true = y_true[mask]
        y_pred = y_pred[mask]

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
```

---

## Exemples d'utilisation

### Via la CLI

```bash
# Mode interactif complet
python main.py predict
# → Sélection dataset (↑↓)
# → Sélection produit (↑↓)
# → Nombre de jours

# Mode direct
python main.py predict -p "Poulet Frais" -d 30 -D enriched --save
```

### Via l'API

```bash
# Prédiction simple
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"product": "Poulet Frais", "days": 30}'
```

### Via Python

```python
from src.data_loader import load_dataset, prepare_prophet_data, train_test_split
from src.model import train_prophet_model, predict_future
from src.metrics import calculate_metrics, print_metrics

# 1. Charger les données
df = load_dataset('data/dataset_stock_hopital_ENRICHI.csv')

# 2. Préparer pour Prophet
prophet_df = prepare_prophet_data(
    df,
    target_column='quantite',
    product_filter='Poulet Frais',
    product_column='nom_produit'
)

# 3. Split train/test
train, test = train_test_split(prophet_df, test_ratio=0.2)

# 4. Entraîner
model = train_prophet_model(
    train,
    weekly_seasonality=True,
    yearly_seasonality=True,
    changepoint_prior_scale=0.05
)

# 5. Prédire
predictions = predict_future(model, periods=30)

# 6. Évaluer
metrics = calculate_metrics(test['y'], model.predict(test)['yhat'])
print_metrics(metrics)
```

---

## 📖 Ressources

- [Documentation officielle Prophet](https://facebook.github.io/prophet/)
- [Paper : Forecasting at Scale](https://peerj.com/preprints/3190/)
- [Guide Dataset Enrichi](../data/GUIDE_DATASET_ENRICHI.md)
- [Configuration des paramètres](../config/model_params.yaml)

---

## 🔗 Liens internes

- [README principal](../README.md)
- [Guide Dataset Enrichi](../data/GUIDE_DATASET_ENRICHI.md)
- [Comparaison des datasets](../data/README_DATASETS.md)
