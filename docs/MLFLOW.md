# MLflow Integration

Cette intégration ajoute un suivi d'expériences simple autour de la commande `predict`, du notebook Prophet et de l'API FastAPI.

## Ce qui est tracé

- paramètres métier: produit, dataset, horizon
- paramètres Prophet issus de la configuration projet
- métriques: `mae`, `mape`, `rmse`, `r2`
- artefacts: `future_predictions.csv`, `test_predictions.csv`, `metrics.json`
- fichiers additionnels du dossier `results/` si `--save` est activé

## Installation

```bash
pip install -r requirements.txt
```

## Démarrage local

```bash
mlflow ui --backend-store-uri "sqlite:///C:/Users/mxmle/project/final-project/mlflow.db" --host 127.0.0.1 --port 5000
```

Interface disponible par défaut sur `http://127.0.0.1:5000`.

Le backend local par défaut du projet est désormais `sqlite:///C:/Users/mxmle/project/final-project/mlflow.db`, ce qui évite l'avertissement de dépréciation du file store local.

- les métadonnées MLflow sont stockées dans `mlflow.db`
- les artefacts des runs courants restent enregistrés localement dans `mlruns/<experiment_id>/`
- les anciens dossiers hérités du file store peuvent être archivés séparément dans `mlruns_legacy_archive/`
- certaines expériences existantes peuvent pointer vers des artefacts dans `notebooks/mlruns/<experiment_id>/` si elles ont été créées depuis le notebook avant harmonisation

Si l'UI semble vide alors que les runs existent, vérifier les points suivants:

- utiliser le chemin absolu dans `--backend-store-uri`
- sélectionner la bonne expérience dans la barre latérale
- recharger complètement la page après démarrage de l'UI

## Utilisation

```bash
python main.py predict --product "Poulet Frais" --days 30 --mlflow
```

Avec un stockage local explicite:

```bash
python main.py predict --product "Poulet Frais" --days 30 --mlflow --mlflow-experiment "hospital-stock-prediction" --mlflow-tracking-uri "sqlite:///C:/Users/mxmle/project/final-project/mlflow.db"
```

La commande `predict` de la CLI accepte aussi:

- `--save` pour joindre les artefacts générés dans `results/`
- `--mlflow-experiment` pour changer le nom d'expérience
- `--mlflow-tracking-uri` pour surcharger le backend par défaut

## Utilisation via API

```bash
curl -X POST "http://localhost:8000/predict" \
	-H "Content-Type: application/json" \
	-d '{"product": "Poulet Frais", "days": 30, "dataset": "enriched", "enable_mlflow": true, "mlflow_experiment": "hospital-stock-api"}'
```

Quand le tracking est activé, la réponse JSON inclut un objet `tracking` avec:

- `enabled`
- `logged`
- `experiment_name`
- `tracking_uri`
- `run_id`
- `run_name`

L'API utilise `hospital-stock-api` comme expérience par défaut.

## Utilisation via notebook

Le notebook `Analyse_Mont_Vert_ENRICHI.ipynb` logge aussi dans MLflow.

- expérience par défaut: `prophet-notebook-enrichi`
- backend par défaut: `sqlite:///C:/Users/mxmle/project/final-project/mlflow.db`
- métriques loggées: `mae`, `mape`, `rmse`, `r2`
- artefacts loggés: prédictions test, prédictions futures, résumé du modèle et exports du dossier `results/`

Le notebook recharge explicitement `src.mlflow_utils` pour éviter les problèmes de cache du kernel Jupyter après modification du code source.

## Extension recommandée

Les étapes suivantes logiques sont:

- une commande de backtesting ou de comparaison de modèles
- un vrai serveur MLflow partagé si plusieurs personnes doivent consulter les runs
