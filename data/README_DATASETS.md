# 📊 Datasets Disponibles

Ce dossier contient 3 versions du dataset de la Clinique du Mont Vert, chacune optimisée pour différents cas d'usage.

---

## 🗂️ Vue d'ensemble

| Dataset | Période | Lignes | Colonnes | Taille | Usage recommandé |
|---------|---------|--------|----------|--------|------------------|
| **dataset_stock_hopital.csv** | 2022-2024 (3 ans) | 51,839 | 15 | 5.6 MB | ✅ **Analyses de base, démo rapide** |
| **dataset_stock_hopital_REALISTE.csv** | 2022-2024 (3 ans) | 24,000 | 15 | 2.7 MB | ⚠️ Version intermédiaire |
| **dataset_stock_hopital_ENRICHI.csv** | 2020-2024 (5 ans) | 85,809 | 22 | 12 MB | 🚀 **Prophet avec regressors avancés** |

---

## 📋 Détails des Datasets

### 1. dataset_stock_hopital.csv (Version de Base)

**Caractéristiques** :
- ✅ Simple et rapide à charger
- ✅ Parfait pour débuter
- ✅ Contient les données essentielles
- ❌ Pas de régresseurs externes
- ❌ Pas d'événements (COVID, holidays)

**Colonnes (15)** :
```
date, type_operation, id_lot, numero_lot, id_produit, nom_produit,
type_produit, unite, quantite, stock_theorique, temperature,
date_expiration, id_fournisseur, nom_fournisseur, id_arrivage
```

**Cas d'usage** :
- Analyses exploratoires rapides
- Démonstration du concept
- Tests de code
- Formation initiale

---

### 2. dataset_stock_hopital_REALISTE.csv (Version Intermédiaire)

**Caractéristiques** :
- ✅ FIFO implémenté
- ✅ Destruction automatique des périmés
- ✅ Plus réaliste que v1
- ⚠️ Moins de lignes (agrégation quotidienne)
- ❌ Pas de régresseurs externes

**Améliorations vs v1** :
- Gestion correcte des lots (FIFO)
- Destruction automatique à expiration
- Moins de gaspillage irréaliste

**Cas d'usage** :
- Analyses avec gestion réaliste des stocks
- Validation du FIFO
- Analyses de gaspillage précises

---

### 3. dataset_stock_hopital_ENRICHI.csv (Version Complète) ⭐

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

| Date | Événement | Impact |
|------|-----------|--------|
| 15/03/2020 | COVID-19 Vague 1 | +50% consommation |
| 01/11/2020 | COVID-19 Vague 2 | +30% consommation |
| 01/05/2021 | Déconfinement | -10% consommation |
| 01/01/2022 | Nouvelle Direction | +10% efficacité |
| 01/09/2023 | Extension Hôpital | +15% capacité |

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

## 🎯 Quel Dataset Choisir ?

### Pour débuter / Tester rapidement
→ **dataset_stock_hopital.csv**
- Charge rapide
- Simple à comprendre
- Suffisant pour les analyses de base

### Pour analyses réalistes de stocks
→ **dataset_stock_hopital_REALISTE.csv**
- FIFO implémenté
- Gestion réaliste du gaspillage
- Analyses précises

### Pour modélisation IA avancée (Prophet)
→ **dataset_stock_hopital_ENRICHI.csv** ⭐
- Tous les régresseurs nécessaires
- Holidays et changepoints intégrés
- Meilleure précision de prédiction
- **RECOMMANDÉ pour votre projet final**

---

## 📚 Documentation Complète

Pour un guide détaillé d'utilisation du dataset enrichi avec Prophet, consultez :
- [GUIDE_DATASET_ENRICHI.md](GUIDE_DATASET_ENRICHI.md) *(à créer)*

## 🔗 Utilisation dans les Notebooks

```python
import pandas as pd

# Chargement du dataset de base
df_base = pd.read_csv('data/dataset_stock_hopital.csv')
df_base['date'] = pd.to_datetime(df_base['date'])

# Chargement du dataset enrichi (recommandé)
df_enrichi = pd.read_csv('data/dataset_stock_hopital_ENRICHI.csv')
df_enrichi['date'] = pd.to_datetime(df_enrichi['date'])

print(f"Base : {len(df_base)} lignes, {len(df_base.columns)} colonnes")
print(f"Enrichi : {len(df_enrichi)} lignes, {len(df_enrichi.columns)} colonnes")
```

---

## 📊 Comparaison des Résultats Attendus

| Métrique | Base | Réaliste | Enrichi |
|----------|------|----------|---------|
| **MAE (Poulet frais)** | ~3.5 kg | ~3.0 kg | **~2.0 kg** |
| **MAPE** | ~35% | ~30% | **~20%** |
| **Précision** | Correcte | Bonne | **Excellente** |
| **Temps calcul** | Rapide | Moyen | Lent |
| **Complexité** | Faible | Moyenne | Élevée |

---

**💡 Recommandation** : Utilisez **dataset_stock_hopital_ENRICHI.csv** pour votre projet final afin d'obtenir les meilleures performances de prédiction !
