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
│   └── dataset_stock_hopital.csv          # Données sources (3 ans de transactions)
├── notebooks/
│   └── Analyse_Mont_Vert_LOCAL_VSCODE.ipynb  # Notebook principal d'analyse
├── results/                               # Résultats des analyses
│   └── [YYYYMMDD_HHMMSS]/               # Un dossier par exécution
│       ├── predictions_*.csv
│       ├── summary_*.json
│       └── graphs/
│           └── *.png
├── .gitignore
├── README.md
└── requirements.txt
```

## Données

Le dataset contient 51 839 transactions sur 3 ans (2022-2024) incluant :
- 40 produits distincts
- 12 fournisseurs
- Types d'opérations : ARRIVAGE et SORTIE
- Informations : quantités, stock théorique, température, dates d'expiration

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
