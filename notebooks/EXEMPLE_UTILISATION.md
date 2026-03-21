# Guide du Notebook Principal

Ce document décrit ce que fait le notebook principal [Analyse_Mont_Vert_ENRICHI.ipynb](c:/Users/mxmle/project/final-project/notebooks/Analyse_Mont_Vert_ENRICHI.ipynb), quelles données il consomme, quelles sorties il produit, et à quoi servent ses différentes étapes.

## Objectif du notebook

Le notebook réalise une analyse complète de prévision de consommation hospitalière sur le dataset enrichi.

Il sert à :

- charger et contrôler le dataset enrichi
- sélectionner un produit hospitalier précis
- préparer une série temporelle quotidienne compatible avec Prophet
- entraîner un modèle Prophet enrichi avec régresseurs externes
- mesurer sa performance sur une période de test
- produire des prévisions futures sur 28 jours
- transformer ces prévisions en recommandation de commande métier
- exporter les résultats dans un dossier horodaté
- journaliser l'analyse dans MLflow
- comparer le résultat du notebook avec l'API FastAPI

## Entrées principales

Le notebook repose sur quelques variables faciles à modifier :

- `FICHIER_CSV` : chemin du dataset enrichi, par défaut `../data/dataset_stock_hopital_ENRICHI.csv`
- `PRODUIT_ANALYSE` : produit étudié, par exemple `Poisson blanc` ou `Poulet frais`
- `MLFLOW_EXPERIMENT` : nom de l'expérience MLflow, par défaut `prophet-notebook-enrichi`
- `COUVERTURE_SECURITE_JOURS` : couverture de sécurité en jours, par défaut `2`

## Déroulé du notebook

### 1. Imports et configuration

Le notebook charge les bibliothèques Python de base, Prophet, les utilitaires MLflow, et le gestionnaire de résultats optionnel `results_manager.py`.

Il configure aussi :

- l'affichage pandas
- le style matplotlib
- le chemin racine du projet
- le backend MLflow SQLite du projet

### 2. Chargement du dataset enrichi

Le notebook ouvre le fichier CSV enrichi contenant les historiques de stock et les variables contextuelles.

Il vérifie :

- le nombre de lignes et de colonnes
- les colonnes de base métier
- les régresseurs externes disponibles

Les variables contextuelles utilisées dans l'analyse sont :

- `temperature`
- `taux_occupation`
- `nb_patients`
- `epidemie_grippe`
- `jour_ferie`
- `covid_impact`

### 3. Préparation des données

Le notebook convertit les dates, puis filtre un seul produit sur les lignes de consommation uniquement.

Ensuite il :

- agrège les quantités par jour
- complète les dates manquantes pour obtenir une série journalière continue
- remplit les jours sans consommation avec `0`
- remplit les régresseurs continus manquants avec leur moyenne
- remplit les régresseurs binaires manquants avec `0`
- renomme les colonnes au format Prophet : `ds` pour la date et `y` pour la cible

### 4. Construction des événements spéciaux

Le notebook prépare un DataFrame `holidays` pour Prophet avec deux types d'événements :

- les jours fériés
- les périodes COVID

Ces événements permettent au modèle de prendre en compte les chocs ou anomalies connus dans la consommation.

### 5. Changepoints manuels

Le notebook ajoute plusieurs dates métier importantes comme changepoints manuels.

L'objectif est de mieux modéliser les changements de tendance liés à des événements structurants, par exemple :

- vagues COVID
- déconfinement
- nouvelle direction
- extension de l'hôpital

### 6. Split train/test

La série temporelle est divisée chronologiquement :

- train : toutes les données sauf la dernière année
- test : la dernière année

Cela permet une évaluation réaliste sur une période récente non vue à l'entraînement.

### 7. Modèle Prophet enrichi

Le notebook instancie un modèle Prophet avancé avec :

- holidays
- saisonnalité annuelle et hebdomadaire
- mode multiplicatif
- changepoints manuels
- intervalle de confiance à 85%

Puis il ajoute 4 régresseurs explicites au modèle :

- `temperature`
- `taux_occupation`
- `nb_patients`
- `epidemie_grippe`

Le modèle est ensuite entraîné sur le jeu d'entraînement.

### 8. Évaluation du modèle

Le notebook prédit la période de test et calcule plusieurs métriques de performance :

- `MAE`
- `MAPE`
- `RMSE`
- plus tard `R2` pour le tracking MLflow et la comparaison API

Le `MAPE` est calculé uniquement sur les jours où la consommation réelle est strictement positive, afin d'éviter les divisions par zéro sur les jours sans consommation.

### 9. Analyse des régresseurs

Le notebook extrait les coefficients des régresseurs avec `regressor_coefficients(model)`.

Cela permet d'interpréter le sens et l'intensité de l'effet de chaque variable contextuelle sur la consommation prévue.

### 10. Prévisions futures sur 28 jours

Le notebook réentraîne un modèle final sur l'ensemble des données historiques, puis construit un horizon futur de 28 jours.

Pour les régresseurs futurs, il applique des hypothèses simples :

- moyenne historique pour `temperature`, `taux_occupation` et `nb_patients`
- activation de `epidemie_grippe` sur les mois d'hiver

Il génère ensuite les prévisions futures avec :

- `yhat`
- `yhat_lower`
- `yhat_upper`

### 11. Recommandation de commande métier

Le notebook convertit la prévision en recommandation opérationnelle.

La logique métier repose sur :

- le stock théorique encore consommable à la dernière date observée
- un horizon de commande de 7 jours
- aucun arrivage futur planifié disponible dans les données
- un stock de sécurité correspondant à 2 jours de consommation moyenne prévue

Les sorties métier produites incluent notamment :

- `stock_disponible_actuel`
- `arrivages_planifies`
- `consommation_prevue_horizon`
- `stock_securite`
- `quantite_a_commander`
- `couverture_estimee_jours`

### 12. Visualisations

Le notebook crée trois graphiques PNG dans un dossier de résultats horodaté :

- un graphique principal de prévision
- un graphique de composantes Prophet
- un graphique des changepoints

### 13. Exports de résultats

Le notebook écrit plusieurs fichiers dans `results/analyse-{produit}-{timestamp}/` :

- `predictions_{produit}_enrichi.png`
- `components_{produit}_enrichi.png`
- `changepoints_{produit}_enrichi.png`
- `predictions_{produit}_28j.csv`
- `summary_{produit}.json`
- `README.md`

Le fichier JSON contient :

- le contexte du dataset
- les métriques du modèle
- la configuration Prophet
- les statistiques de prévision
- la recommandation de commande
- puis les informations de tracking MLflow une fois la cellule MLflow exécutée

### 14. Tracking MLflow

Le notebook peut enregistrer l'analyse dans MLflow si le package est disponible dans l'environnement.

Le tracking journalise :

- les métriques du modèle
- les paramètres Prophet
- les tags du notebook
- les prédictions de test et futures
- le modèle Prophet entraîné
- les artefacts exportés dans le dossier de résultats

Le résumé JSON est ensuite mis à jour avec un bloc `tracking` contenant par exemple :

- `enabled`
- `logged`
- `experiment_name`
- `tracking_uri`
- `run_id`
- `run_name`
- `model_logged`

### 15. Comparaison notebook vs API

La dernière cellule compare le comportement du notebook avec l'API FastAPI du projet.

Elle :

- recharge `api.main` et `src.enriched_pipeline`
- appelle `/predict` via `TestClient`
- compare les prédictions futures du notebook et celles de l'API
- compare `mae`, `mape`, `rmse`, `r2`
- compare aussi la recommandation métier

Cette cellule sert à vérifier que la logique API et la logique notebook restent alignées.

## Fichiers produits par une exécution

Une exécution complète du notebook principal produit généralement :

- des sorties console dans les cellules
- un dossier horodaté dans `results/`
- trois graphiques PNG
- un CSV de prévision sur 28 jours
- un JSON de synthèse enrichi
- un README local au run
- éventuellement un run MLflow avec artefacts et modèle

## Ce que le notebook ne fait pas

Le notebook principal ne fait pas directement :

- de sélection interactive de produit dans l'interface
- de backtesting multi-produits automatique
- d'optimisation automatique des hyperparamètres
- de gestion d'arrivages futurs planifiés réels

Il s'agit avant tout d'un notebook d'analyse avancée, de validation métier et de génération d'artefacts.

## Quand utiliser ce notebook

Il est utile si vous voulez :

- analyser un produit en détail
- comprendre l'effet des régresseurs externes
- produire des graphes et exports propres
- obtenir une recommandation de commande justifiée
- tracer l'analyse dans MLflow
- vérifier l'alignement entre notebook et API

## Utilisation recommandée

1. Ouvrir [Analyse_Mont_Vert_ENRICHI.ipynb](c:/Users/mxmle/project/final-project/notebooks/Analyse_Mont_Vert_ENRICHI.ipynb).
2. Modifier `PRODUIT_ANALYSE` si nécessaire.
3. Exécuter les cellules dans l'ordre.
4. Vérifier les métriques obtenues sur le jeu de test.
5. Consulter les exports générés dans `results/`.
6. Si MLflow est actif, vérifier le run associé dans l'interface MLflow.

## En résumé

Le notebook principal est la version analytique de référence du projet.

Il combine dans un seul flux :

- préparation des données enrichies
- modélisation Prophet avancée
- évaluation quantitative
- interprétation des régresseurs
- prévision future
- recommandation métier de commande
- export de livrables
- tracking MLflow
- vérification de parité avec l'API
