# 📊 Dataset Disponible

Ce dossier conserve uniquement le dataset enrichi de la Clinique du Mont Vert, qui sert de référence pour le notebook principal, l'API et le tracking MLflow.

---

## 🗂️ Vue d'ensemble

| Dataset                               | Période           | Lignes | Colonnes | Taille | Usage recommandé                         |
| ------------------------------------- | ----------------- | ------ | -------- | ------ | ---------------------------------------- |
| **dataset_stock_hopital_ENRICHI.csv** | 2020-2024 (5 ans) | 85,809 | 22       | 12 MB  | 🚀 **Prophet avec régressseurs avancés** |

---

## 📋 Détails du Dataset

### dataset_stock_hopital_ENRICHI.csv ⭐

**Caractéristiques** :

- ✅ **5 ans de données** (2020-2024)
- ✅ **22 colonnes** avec 7 régresseurs externes
- ✅ **Holidays** intégrés (jours fériés français)
- ✅ **Changepoints** marqués (COVID, extensions)
- ✅ **Saisonnalité renforcée**
- ✅ Optimisé pour Prophet

**Colonnes (22)** :

**Colonnes de base (15)** :

```
date, id_produit, nom_produit, type_produit, type_operation,
type_sortie, quantite, unite, id_lot, id_arrivage,
id_fournisseur, nom_fournisseur, date_expiration,
stock_theorique, temperature_stockage
```

**Régresseurs externes (7) ✨** :

```
temperature          : Température extérieure (°C)
taux_occupation      : Taux d'occupation hôpital (%)
nb_patients          : Nombre de patients
epidemie_grippe      : 1 = épidémie, 0 = non
vacances_scolaires   : 1 = vacances, 0 = non
jour_ferie           : 1 = férié, 0 = non
covid_impact         : 1 = période COVID, 0 = non
```

**Événements majeurs intégrés** :

| Date       | Événement          | Impact            |
| ---------- | ------------------ | ----------------- |
| 15/03/2020 | COVID-19 Vague 1   | +50% consommation |
| 01/11/2020 | COVID-19 Vague 2   | +30% consommation |
| 01/05/2021 | Déconfinement      | -10% consommation |
| 01/01/2022 | Nouvelle Direction | +10% efficacité   |
| 01/09/2023 | Extension Hôpital  | +15% capacité     |

**Statistiques** :

```
📊 Période           : 2020-01-01 → 2024-12-31 (5 ans)
📊 Total opérations  : 85,809 enregistrements
📊 Nombre de produits: 40

Opérations :
├─ ENTREES        : 13,972 (16.3%)
└─ SORTIES        : 71,837 (83.7%)
    ├─ CONSOMMATION : 62,576 (87%)
    └─ DESTRUCTION  : 9,261 (13%)

Gaspillage :
├─ Total détruit  : 150,962 unités
├─ Total consommé : 733,165 unités
└─ Taux           : 17.07%

Régresseurs (moyennes) :
├─ Température    : 15.0°C
├─ Occupation     : 77.1%
├─ Patients       : 192/jour
├─ Jours grippe   : 47% des jours en hiver
├─ Jours vacances : 11.5% de l'année
├─ Jours fériés   : 360/an
└─ Jours COVID    : 14 mois cumulés
```

**Cas d'usage** :

- 🚀 **Modélisation avancée avec Prophet**
- Analyse d'impact des événements (COVID)
- Prédictions avec variables externes
- Projets de Master/Recherche
- Démonstration complète de l'IA

---

## 🎯 Pourquoi ce dataset

Le projet a été recentré sur ce seul fichier parce qu'il contient tout ce qui est nécessaire pour la modélisation finale :

- tous les régresseurs utiles à Prophet
- les variables d'événements métier et contextuels
- l'historique long 2020-2024
- le meilleur support pour l'analyse du notebook et l'API

---

## 📚 Documentation Complète

Pour un guide détaillé d'utilisation du dataset enrichi avec Prophet, consultez :

- [GUIDE_DATASET_ENRICHI.md](GUIDE_DATASET_ENRICHI.md)

## 🔗 Utilisation dans les Notebooks

```python
import pandas as pd

# Chargement du dataset enrichi
df_enrichi = pd.read_csv('data/dataset_stock_hopital_ENRICHI.csv')
df_enrichi['date'] = pd.to_datetime(df_enrichi['date'])

print(f"Enrichi : {len(df_enrichi)} lignes, {len(df_enrichi.columns)} colonnes")
```

---

**💡 Recommandation** : Utilisez **dataset_stock_hopital_ENRICHI.csv** comme unique source de vérité du projet.
