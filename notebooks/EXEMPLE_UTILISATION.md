# Comment utiliser le système de gestion des résultats

## Problème résolu

Chaque fois que vous exécutez le notebook d'analyse, les résultats (graphiques, CSV, JSON) étaient écrasés. Maintenant, **chaque exécution est sauvegardée dans un dossier horodaté unique**.

## Structure des résultats

```
results/
├── 20250112_103045/          # Exécution du 12 jan 2025 à 10:30:45
│   ├── graphs/
│   │   ├── graph_top_10_produits.png
│   │   ├── graph_predictions_poulet_frais.png
│   │   └── ...
│   ├── predictions_poulet_frais_4semaines.csv
│   ├── summary_poulet_frais.json
│   └── README.txt            # Résumé automatique de cette exécution
├── 20250112_150230/          # Exécution du 12 jan 2025 à 15:02:30
│   └── ...
└── 20250113_091520/          # Exécution du 13 jan 2025 à 09:15:20
    └── ...
```

## Utilisation dans votre notebook

### Méthode 1 : Ajouter au début du notebook (après les imports)

```python
# Importer le gestionnaire de résultats
from results_manager import ResultsManager

# Créer le gestionnaire et préparer le dossier
results_mgr = ResultsManager()
results_mgr.create_run_directory()

print(f"📁 Résultats seront sauvegardés dans : {results_mgr.get_run_path()}")
```

### Méthode 2 : Modifier les sauvegardes dans le notebook

**AVANT (ancienne méthode) :**
```python
# Étape 5 : Sauvegarder les graphiques
plt.savefig('graph_top_10_produits.png', dpi=300, bbox_inches='tight')
```

**APRÈS (nouvelle méthode) :**
```python
# Étape 5 : Sauvegarder les graphiques
filename = f'{results_mgr.get_run_path()}/graphs/graph_top_10_produits.png'
plt.savefig(filename, dpi=300, bbox_inches='tight')
```

OU encore plus simple :
```python
# Sauvegarder d'abord localement
plt.savefig('graph_top_10_produits.png', dpi=300, bbox_inches='tight')

# Puis copier dans le dossier de résultats
results_mgr.save_graph('graph_top_10_produits.png')
```

### Méthode 3 : Sauvegarder tous les fichiers à la fin

```python
# À la fin du notebook (Étape 9)

# Sauvegarder tous les graphiques
for graph_file in ['graph_top_10_produits.png',
                   'graph_produits_perissables.png',
                   'graph_saisonnalite_mensuelle.png',
                   'graph_pattern_hebdomadaire.png',
                   f'graph_evolution_{PRODUIT_ANALYSE.replace(" ", "_").lower()}.png',
                   f'graph_predictions_{PRODUIT_ANALYSE.replace(" ", "_").lower()}.png']:
    if Path(graph_file).exists():
        results_mgr.save_graph(graph_file)

# Sauvegarder les données
results_mgr.save_data(f'predictions_{PRODUIT_ANALYSE.replace(" ", "_").lower()}_4semaines.csv')
results_mgr.save_data(f'summary_{PRODUIT_ANALYSE.replace(" ", "_").lower()}.json')

# Créer un fichier résumé
results_mgr.create_summary_file(
    product_name=PRODUIT_ANALYSE,
    summary_dict={
        "Consommation moyenne": f"{moyenne_jour:.2f} kg/jour",
        "Durée de vie": f"{duree_vie_moyenne:.1f} jours",
        "Prédictions": f"{len(predictions_futures)} jours",
        "MAE": f"{mae:.2f} kg" if mae else "N/A"
    }
)

print(f"\n🎉 Tous les résultats ont été sauvegardés dans : {results_mgr.get_run_path()}")
```

## Voir les exécutions précédentes

Ajoutez ceci au début de votre notebook pour voir l'historique :

```python
from results_manager import ResultsManager

# Créer le gestionnaire
results_mgr = ResultsManager()

# Afficher les 10 dernières exécutions
results_mgr.print_previous_runs(limit=10)

# Créer un nouveau dossier pour cette exécution
results_mgr.create_run_directory()
```

## Avantages

✅ **Historique complet** : Gardez toutes vos analyses
✅ **Comparaison facile** : Comparez les résultats entre différentes exécutions
✅ **Reproductibilité** : Chaque analyse est datée et documentée
✅ **Sécurité** : Plus de risque d'écraser des résultats importants
✅ **Organisation** : Structure claire et professionnelle

## Retrouver une ancienne analyse

Les dossiers utilisent le format `YYYYMMDD_HHMMSS` :
- `20250112_103045` = 12 janvier 2025 à 10h30:45
- `20250215_143020` = 15 février 2025 à 14h30:20

Chaque dossier contient un fichier `README.txt` avec un résumé de l'analyse.

## Nettoyage

Pour supprimer les anciennes exécutions, il suffit de supprimer les dossiers correspondants dans `results/`.

## Exemple complet

Voir le notebook [Analyse_Mont_Vert_LOCAL_VSCODE.ipynb](Analyse_Mont_Vert_LOCAL_VSCODE.ipynb) pour un exemple d'intégration complète.
