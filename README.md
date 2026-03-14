# 🏥 Hospital Stock Management AI

**Clinique du Mont Vert** - Système de prédiction de stock hospitalier avec IA

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![Prophet](https://img.shields.io/badge/Prophet-1.1+-orange.svg)](https://facebook.github.io/prophet/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📋 Description

Projet de Master EISI utilisant l'intelligence artificielle pour prédire la consommation de produits hospitaliers et optimiser la gestion des stocks. L'objectif est de réduire le gaspillage alimentaire et prévenir les ruptures de stock.

Le projet couvre désormais deux usages complémentaires:

- une API FastAPI pour exposer les prédictions et le tracking MLflow
- un notebook enrichi Prophet qui produit aussi une recommandation opérationnelle de commande

### 🎯 Résultats attendus

- ✅ Réduction de **30%** du gaspillage alimentaire
- ✅ Réduction de **50%** des ruptures de stock
- ✅ Économies annuelles estimées à **60 000€**

## 🏗️ Structure du Projet

```
hospital-stock-management-ai/
├── 📁 api/                    # API REST FastAPI
│   ├── __init__.py
│   ├── main.py               # Application FastAPI
│   └── models.py             # Modèles Pydantic
├── 📁 config/                 # Configuration centralisée
│   ├── __init__.py
│   ├── settings.py           # Settings Python
│   ├── products.yaml         # Définition des produits
│   └── model_params.yaml     # Paramètres Prophet
├── 📁 data/                   # Datasets
│   ├── dataset_stock_hopital_ENRICHI.csv
│   ├── README_DATASETS.md    # Description du dataset enrichi
│   └── GUIDE_DATASET_ENRICHI.md
├── 📁 notebooks/              # Analyses Jupyter
│   ├── Analyse_Mont_Vert_LOCAL_VSCODE.ipynb
│   ├── Analyse_Mont_Vert_ENRICHI.ipynb
│   └── results_manager.py
├── 📁 results/                # Résultats générés
│   └── analyse-{product}-{timestamp}/
├── 📁 src/                    # Code source modulaire
│   ├── __init__.py
│   ├── data_loader.py        # Chargement données
│   ├── model.py              # Modèle Prophet
│   ├── metrics.py            # Calcul métriques
│   ├── mlflow_utils.py       # Tracking MLflow partagé
│   └── visualization.py      # Graphiques
├── 📁 tests/                  # Tests unitaires
│   ├── conftest.py           # Fixtures pytest
│   ├── test_data_loader.py
│   ├── test_metrics.py
│   └── test_config.py
├── Dockerfile                 # Image Docker
├── docker-compose.yml         # Orchestration
├── requirements.txt           # Dépendances
├── pytest.ini                 # Config pytest
└── LICENSE                    # MIT License
```

## 📊 Données

### Dataset disponible

| Dataset        | Période   | Lignes | Colonnes | Usage                     |
| -------------- | --------- | ------ | -------- | ------------------------- |
| **Enrichi** ⭐ | 2020-2024 | 85,809 | 22       | **Prophet + régresseurs** |

Le projet utilise désormais uniquement le dataset enrichi avec variables contextuelles et historique 2020-2024 pour les analyses Prophet avancées.

📚 Documentation : [data/README_DATASETS.md](data/README_DATASETS.md)

## 🔮 Modèle de Prédiction

Ce projet utilise **Prophet** (Meta/Facebook) pour la prédiction de séries temporelles. Prophet est particulièrement adapté car il gère automatiquement :

- 📅 **Saisonnalité multiple** : patterns hebdomadaires et annuels
- 🌡️ **Régresseurs externes** : température, occupation, épidémies...
- 📊 **Données manquantes** : jours sans consommation
- 📈 **Changements de tendance** : impact COVID, nouvelles pratiques

📚 Documentation complète : [docs/PROPHET_MODEL.md](docs/PROPHET_MODEL.md)

## 🚀 Installation

### Prérequis

- Python 3.11+
- pip ou conda

### Installation rapide

```bash
# Cloner le repository
git clone https://github.com/YzRequiem/hospital-stock-management-ai.git
cd hospital-stock-management-ai

# Créer l'environnement virtuel
python -m venv venv

# Activer (Windows)
venv\Scripts\activate

# Activer (Linux/Mac)
source venv/bin/activate

# Installer les dépendances
pip install -r requirements.txt
```

## 💻 Utilisation

### API REST

```bash
# Lancer l'API
uvicorn api.main:app --reload --port 8000

# Documentation Swagger
# http://localhost:8000/docs
```

**Endpoints principaux :**

| Méthode | Endpoint             | Description            |
| ------- | -------------------- | ---------------------- |
| GET     | `/health`            | Health check           |
| GET     | `/products`          | Liste des produits     |
| POST    | `/predict`           | Prédiction (JSON body) |
| GET     | `/predict/{product}` | Prédiction rapide      |

**Exemple d'appel :**

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"product": "Poulet Frais", "days": 30}'
```

**Exemple avec tracking MLflow :**

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"product": "Poulet Frais", "days": 30, "enable_mlflow": true, "mlflow_experiment": "hospital-stock-api"}'
```

La réponse contient alors un bloc `tracking` avec le statut du logging, le `run_id`, le `run_name` et le `tracking_uri`.

### Docker

```bash
# Build et lancer
docker-compose up -d api

# API disponible sur http://localhost:8000

# Lancer les tests
docker-compose --profile test run tests

# Mode développement (hot-reload)
docker-compose --profile dev up api-dev
```

### Notebooks Jupyter

```bash
# Ouvrir dans VS Code
code notebooks/Analyse_Mont_Vert_ENRICHI.ipynb

# Ou lancer Jupyter
jupyter notebook
```

Le notebook enrichi principal couvre:

- entraînement Prophet avec régresseurs externes
- évaluation sur jeu de test
- prédictions futures sur 28 jours
- recommandation de commande sur 7 jours avec stock disponible, stock de sécurité et hypothèse d'absence d'arrivage futur planifié
- export des résultats dans `results/`
- tracking MLflow dans l'expérience `prophet-notebook-enrichi`

### MLflow

Le projet peut tracer les expériences Prophet via le notebook et l'API FastAPI.

```bash
# Lancer l'interface locale avec le backend SQLite du projet
mlflow ui --backend-store-uri "sqlite:///C:/Users/mxmle/project/final-project/mlflow.db" --host 127.0.0.1 --port 5000

# Via l'API FastAPI
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"product": "Poulet Frais", "days": 30, "dataset": "enriched", "enable_mlflow": true, "mlflow_experiment": "hospital-stock-api"}'
```

Interface disponible sur `http://127.0.0.1:5000`.

Par défaut, le backend de tracking est maintenant `sqlite:///C:/Users/mxmle/project/final-project/mlflow.db`, ce qui évite le file store local déprécié. Chaque run enregistre:

- les paramètres métier et Prophet
- les métriques `mae`, `mape`, `rmse`, `r2`
- les prédictions de test et futures
- les artefacts générés dans `results/`

L'API et le notebook n'utilisent pas forcément la même expérience:

- API: `hospital-stock-api` par défaut
- notebook enrichi: `prophet-notebook-enrichi`

## 🧪 Tests

```bash
# Exécuter tous les tests
pytest

# Tests avec couverture
pytest --cov=src --cov-report=html

# Tests verbeux
pytest -v
```

## 📈 Métriques de Performance

Le modèle Prophet est évalué avec :

| Métrique | Description               | Seuil acceptable |
| -------- | ------------------------- | ---------------- |
| **MAE**  | Erreur absolue moyenne    | < 10 kg          |
| **MAPE** | Erreur relative moyenne   | < 25%            |
| **RMSE** | Erreur quadratique        | < 15 kg          |
| **R²**   | Coefficient détermination | > 0.7            |

**Interprétation MAPE :**

- ✅ Excellent : < 10%
- ✅ Très bon : < 15%
- ✅ Bon : < 25%
- ⚠️ Acceptable : < 50%
- ❌ Insuffisant : > 50%

## 🛠️ Technologies

| Catégorie               | Technologies                       |
| ----------------------- | ---------------------------------- |
| **Langage**             | Python 3.11+                       |
| **Data Science**        | pandas, numpy, matplotlib, seaborn |
| **ML/Prédiction**       | Prophet (Facebook)                 |
| **Experiment Tracking** | MLflow                             |
| **API**                 | FastAPI, Pydantic, uvicorn         |
| **Tests**               | pytest, pytest-cov                 |
| **Config**              | YAML, dataclasses                  |
| **Container**           | Docker, docker-compose             |

## 📁 Modules

### `src/data_loader.py`

- `load_dataset()` - Chargement CSV avec encodage auto
- `prepare_prophet_data()` - Préparation format Prophet
- `train_test_split()` - Split chronologique

### `src/model.py`

- `train_prophet_model()` - Entraînement avec saisonnalité
- `predict()` - Prédictions sur données existantes
- `predict_future()` - Prédictions futures

### `src/mlflow_utils.py`

- `get_default_tracking_uri()` - Backend SQLite par défaut du projet
- `check_mlflow_available()` - Vérification disponibilité MLflow
- `log_prediction_run()` - Logging partagé API et notebook

### `src/metrics.py`

- `calculate_mape()` - MAPE avec gestion des zéros
- `calculate_metrics()` - MAE, MAPE, RMSE, R²
- `interpret_mape()` - Interprétation qualitative

### `src/visualization.py`

- `plot_predictions()` - Graphique prédictions vs réel
- `plot_seasonality()` - Composantes saisonnières
- `plot_weekly_pattern()` - Pattern hebdomadaire

## 👥 Auteur

**Master EISI** - Clinique du Mont Vert

## 📄 Licence

Ce projet est sous licence MIT - voir le fichier [LICENSE](LICENSE) pour plus de détails.
