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
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Interface disponible par défaut sur `http://127.0.0.1:5000`.

Le backend local par défaut du projet est désormais `sqlite:///mlflow.db`, ce qui évite l'avertissement de dépréciation du file store local.

- les métadonnées MLflow sont stockées dans `mlflow.db`
- les artefacts des runs courants restent enregistrés localement dans `mlruns/<experiment_id>/`
- les anciens dossiers hérités du file store peuvent être archivés séparément dans `mlruns_legacy_archive/`

## Utilisation

```bash
python main.py predict --product "Poulet Frais" --days 30 --mlflow
```

Avec un stockage local explicite:

```bash
python main.py predict --product "Poulet Frais" --days 30 --mlflow --mlflow-experiment "hospital-stock-prediction" --mlflow-tracking-uri "sqlite:///mlflow.db"
```

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

## Extension recommandée

Les étapes suivantes logiques sont:

- une commande de backtesting ou de comparaison de modèles
- un vrai serveur MLflow partagé si plusieurs personnes doivent consulter les runs
