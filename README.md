# Gestion Intelligente des Stocks Hospitaliers - IA Prédictive

Projet de Master EISI - Analyse prédictive pour l'optimisation de la gestion des stocks de la Clinique du Mont Vert.

## Description

Ce projet utilise des techniques d'intelligence artificielle et d'apprentissage automatique pour prédire la consommation de produits hospitaliers et optimiser la gestion des stocks. L'objectif est de réduire le gaspillage alimentaire et prévenir les ruptures de stock.

### Résultats attendus
- Réduction de 30% du gaspillage alimentaire
- Réduction de 50% des ruptures de stock
- Économies annuelles estimées à 60 000€

## Structure du Projet

```
final-project/
├── data/
│   ├── dataset_stock_hopital.csv          # Dataset de base (3 ans)
│   ├── dataset_stock_hopital_REALISTE.csv # Dataset avec FIFO
│   ├── dataset_stock_hopital_ENRICHI.csv  # ⭐ Dataset enrichi (5 ans + regressors)
│   ├── README_DATASETS.md                 # Comparaison des datasets
│   └── GUIDE_DATASET_ENRICHI.md          # Guide d'utilisation complet
├── notebooks/
│   ├── Analyse_Mont_Vert_LOCAL_VSCODE.ipynb  # Notebook principal
│   ├── results_manager.py                    # Gestionnaire de résultats
│   └── EXEMPLE_UTILISATION.md               # Guide du results manager
├── results/                               # Résultats automatiques
│   └── [YYYYMMDD_HHMMSS]/               # Un dossier par exécution
│       ├── predictions_*.csv
│       ├── summary_*.json
│       ├── README.txt                   # Résumé auto
│       └── graphs/
│           └── *.png
├── .gitignore
├── README.md
└── requirements.txt
```

## Données

### 3 Datasets disponibles

Le projet inclut **3 versions** du dataset, chacune optimisée pour différents cas d'usage :

| Dataset | Période | Lignes | Colonnes | Usage |
|---------|---------|--------|----------|-------|
| **Base** | 2022-2024 | 51,839 | 15 | Analyses de base |
| **Réaliste** | 2022-2024 | 24,000 | 15 | FIFO + gestion réaliste |
| **Enrichi** ⭐ | 2020-2024 | 85,809 | 22 | **Prophet + regressors avancés** |

**Recommandé** : Utilisez le dataset enrichi pour obtenir les meilleures performances de prédiction !

📚 **Documentation détaillée** : Consultez [data/README_DATASETS.md](data/README_DATASETS.md)

### Dataset Enrichi (recommandé)

Le dataset enrichi v3.0 contient **85 809 transactions sur 5 ans** (2020-2024) avec :
- **40 produits** distincts
- **12 fournisseurs**
- **7 régresseurs externes** : température, occupation, patients, épidémies, etc.
- **Holidays intégrés** : jours fériés français, vacances scolaires, COVID
- **Changepoints** : événements majeurs (COVID, extensions)
- **Saisonnalité renforcée** : patterns hebdomadaires et annuels marqués

🚀 **Guide complet** : [data/GUIDE_DATASET_ENRICHI.md](data/GUIDE_DATASET_ENRICHI.md)

## Installation

### Prérequis
- Python 3.8 ou supérieur
- pip

### Configuration

1. Cloner le repository :
```bash
git clone <votre-repo-url>
cd final-project
```

2. Créer un environnement virtuel :
```bash
python -m venv venv
```

3. Activer l'environnement virtuel :
- Windows :
```bash
venv\Scripts\activate
```
- Linux/Mac :
```bash
source venv/bin/activate
```

4. Installer les dépendances :
```bash
pip install -r requirements.txt
```

## Utilisation

1. Ouvrir le notebook dans VS Code ou Jupyter :
```bash
code notebooks/Analyse_Mont_Vert_LOCAL_VSCODE.ipynb
```

2. Exécuter les cellules du notebook dans l'ordre

3. Les résultats seront automatiquement sauvegardés dans `results/[YYYYMMDD_HHMMSS]/`

### Personnalisation

Pour analyser un autre produit, modifier la variable `PRODUIT_ANALYSE` dans la section 6 du notebook.

## Fonctionnalités

### Analyse des Données
- Exploration et nettoyage des données
- Détection des patterns saisonniers
- Identification des produits critiques
- Analyse des produits périssables

### Modélisation Prédictive
- Utilisation de Facebook Prophet pour les prévisions
- Prédictions avec intervalles de confiance
- Prise en compte de la saisonnalité et des tendances

### Visualisations
- Top 10 des produits les plus consommés
- Analyse des produits périssables
- Patterns hebdomadaires et mensuels
- Graphiques de prédictions

### Recommandations Business
- Alertes de gaspillage potentiel
- Suggestions de commandes optimisées
- Analyse des risques de rupture

## Technologies Utilisées

- **Python** : Langage principal
- **pandas** : Manipulation de données
- **numpy** : Calculs numériques
- **matplotlib/seaborn** : Visualisations
- **Prophet** : Prévisions de séries temporelles
- **Jupyter** : Environnement d'analyse interactive

## Résultats

Chaque exécution du notebook génère :
- **Prédictions CSV** : Prévisions quotidiennes sur 4 semaines
- **Résumé JSON** : Statistiques et recommandations
- **Graphiques PNG** : 6 visualisations professionnelles

## Auteur

Projet de Master EISI - Clinique du Mont Vert

## Licence

Projet académique
