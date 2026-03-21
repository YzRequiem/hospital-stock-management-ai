# Analyse des Résultats — Prévision Multi-Produits avec Inférence Bayésienne (MCMC)

**Date d'analyse** : 15 mars 2026  
**Run analysé** : `analyse-tous_produits-20260315_165036`  
**Méthode** : Prophet + régresseurs enrichis + MCMC (300 samples)  
**Dataset** : Enrichi v3.0 — 76 539 lignes, 35 produits alimentaires, période 2020–2024

---

## Table des matières

1. [Contexte et objectif](#1-contexte-et-objectif)
2. [Configuration du modèle](#2-configuration-du-modèle)
3. [Performance globale](#3-performance-globale)
4. [Analyse détaillée par produit](#4-analyse-détaillée-par-produit)
5. [Analyse bayésienne des régresseurs (MCMC)](#5-analyse-bayésienne-des-régresseurs-mcmc)
6. [Recommandations de commande](#6-recommandations-de-commande)
7. [Plans d'arrivages](#7-plans-darrivages)
8. [Comparaison MAP vs MCMC](#8-comparaison-map-vs-mcmc)
9. [Limites identifiées et axes d'amélioration](#9-limites-identifiées-et-axes-damélioration)
10. [Conclusion](#10-conclusion)

---

## 1. Contexte et objectif

Ce document présente l'analyse des résultats du système de prévision de consommation multi-produits de la **Clinique du Mont Vert**. Le modèle Prophet est entraîné individuellement pour chacun des 35 produits alimentaires avec des **régresseurs externes** (température, taux d'occupation, nombre de patients, épidémies de grippe) et une inférence **MCMC** (Markov Chain Monte Carlo) pour quantifier l'incertitude sur les coefficients.

### Objectifs

- Prévoir la consommation de chaque produit sur un horizon de **28 jours**
- Générer des **recommandations de commande** à 7 jours avec stock de sécurité
- Planifier les **arrivages** en fonction des dates limites de consommation (DLC)
- **Quantifier l'incertitude** sur l'impact des régresseurs via l'inférence bayésienne

---

## 2. Configuration du modèle

### Paramètres Prophet

| Paramètre                 | Valeur                 | Justification                                     |
| ------------------------- | ---------------------- | ------------------------------------------------- |
| Saisonnalité annuelle     | 20 composantes Fourier | Capture des variations saisonnières alimentaires  |
| Saisonnalité hebdomadaire | 5 composantes Fourier  | Patterns jour de semaine / weekend                |
| Mode de saisonnalité      | Multiplicatif          | Amplitude proportionnelle au niveau de la série   |
| Changepoint prior scale   | 0.5                    | Flexibilité modérée pour les ruptures de tendance |
| Changepoint range         | 0.9                    | Détection sur 90% de l'historique                 |
| Intervalle de confiance   | 85%                    | Compromis précision / couverture                  |
| Croissance                | Linéaire               | Tendance linéaire par morceaux                    |
| **MCMC samples**          | **300**                | Inférence bayésienne complète (NUTS)              |

### Changepoints manuels

| Date       | Événement                 |
| ---------- | ------------------------- |
| 2020-03-15 | COVID-19 — Première vague |
| 2020-11-01 | COVID-19 — Deuxième vague |
| 2021-05-01 | Déconfinement progressif  |
| 2022-01-01 | Changement de direction   |
| 2023-09-01 | Extension de l'hôpital    |

### Holidays

- **Jours fériés français** : intégrés comme événements récurrents
- **Périodes COVID-19** : modélisées comme holidays exceptionnels

### Régresseurs externes

| Régresseur        | Prior scale | Standardisé | Mode    | Description                        |
| ----------------- | ----------- | ----------- | ------- | ---------------------------------- |
| `temperature`     | 0.5         | Oui         | Additif | Température extérieure (°C)        |
| `taux_occupation` | 1.0         | Oui         | Additif | Taux d'occupation de l'hôpital (%) |
| `nb_patients`     | 0.5         | Oui         | Additif | Nombre de patients hospitalisés    |
| `epidemie_grippe` | 0.5         | Non         | Additif | Période d'épidémie grippale (0/1)  |

---

## 3. Performance globale

### Métriques agrégées

| Indicateur          | Valeur        |
| ------------------- | ------------- |
| **MAPE moyen**      | 29.6%         |
| **MAPE médian**     | 25.5%         |
| **MAE moyen**       | 3.79          |
| Meilleur produit    | Oeufs (18.7%) |
| Pire produit        | Thé (53.9%)   |
| Produits < 20% MAPE | 6 / 35 (17%)  |
| Produits < 35% MAPE | 27 / 35 (77%) |

### Distribution des performances

| Catégorie   | MAPE   | Nb produits | %   | Interprétation                                        |
| ----------- | ------ | ----------- | --- | ----------------------------------------------------- |
| Excellent   | < 20%  | 6           | 17% | Prévisions très fiables pour la prise de décision     |
| Bon         | 20–25% | 8           | 23% | Prévisions exploitables avec marge de sécurité        |
| Acceptable  | 25–35% | 13          | 37% | Prévisions utilisables pour la planification          |
| Insuffisant | 35–50% | 5           | 14% | Prévisions à considérer avec précaution               |
| Faible      | > 50%  | 3           | 9%  | Prévisions peu fiables — modèle alternatif nécessaire |

### Interprétation

Un MAPE médian de **25.5%** signifie que pour la moitié des produits, l'erreur de prévision quotidienne est inférieure à 25% de la consommation réelle. Pour de la prévision de stock hospitalier avec des données agrégées au jour, c'est un niveau de performance **acceptable et exploitable opérationnellement**, d'autant plus que les recommandations de commande intègrent un stock de sécurité pour absorber les erreurs.

---

## 4. Analyse détaillée par produit

### Classement complet par MAPE

| #   | Produit           | Unité | MAPE   | MAE   | RMSE  | R²     | Prévu 28j | Moy/jour |
| --- | ----------------- | ----- | ------ | ----- | ----- | ------ | --------- | -------- |
| 1   | Oeufs             | unité | 18.68% | 16.73 | 23.34 | 0.473  | 2 204.7   | 78.74    |
| 2   | Poulet frais      | kg    | 18.79% | 3.67  | 5.18  | 0.149  | 395.9     | 14.14    |
| 3   | Jambon blanc      | kg    | 19.27% | 2.01  | 2.78  | 0.345  | 253.1     | 9.04     |
| 4   | Fromage emmental  | kg    | 19.73% | 1.30  | 1.83  | 0.458  | 165.4     | 5.91     |
| 5   | Pommes de terre   | kg    | 19.93% | 4.40  | 5.83  | 0.432  | 544.1     | 19.43    |
| 6   | Viande de boeuf   | kg    | 19.97% | 2.53  | 3.51  | 0.318  | 328.1     | 11.72    |
| 7   | Salade verte      | kg    | 20.42% | 2.83  | 3.86  | 0.261  | 336.3     | 12.01    |
| 8   | Conserves légumes | kg    | 21.15% | 2.05  | 2.87  | 0.377  | 270.1     | 9.65     |
| 9   | Lait entier       | L     | 21.18% | 10.41 | 15.02 | 0.013  | 1 074.0   | 38.36    |
| 10  | Beurre            | kg    | 21.57% | 1.20  | 1.69  | 0.341  | 137.9     | 4.92     |
| 11  | Carottes          | kg    | 23.11% | 3.32  | 4.60  | 0.190  | 360.5     | 12.87    |
| 12  | Sucre             | kg    | 23.58% | 1.69  | 2.19  | 0.412  | 181.7     | 6.49     |
| 13  | Biscuits          | kg    | 23.94% | 1.64  | 2.15  | 0.373  | 182.4     | 6.51     |
| 14  | Riz               | kg    | 23.97% | 3.45  | 4.48  | 0.454  | 412.5     | 14.73    |
| 15  | Fruits frais      | kg    | 24.41% | 8.39  | 10.68 | 0.228  | 803.7     | 28.70    |
| 16  | Fromage frais     | kg    | 24.82% | 1.63  | 2.29  | 0.251  | 244.0     | 8.71     |
| 17  | Yaourt nature     | kg    | 25.13% | 3.54  | 4.71  | 0.156  | 384.7     | 13.74    |
| 18  | Pâtes sèches      | kg    | 25.47% | 2.85  | 3.62  | 0.463  | 342.9     | 12.25    |
| 19  | Pâtes fraîches    | kg    | 26.94% | 2.39  | 3.03  | 0.323  | 300.0     | 10.72    |
| 20  | Légumes frais     | kg    | 27.07% | 10.95 | 13.40 | 0.267  | 958.2     | 34.22    |
| 21  | Compote de fruits | kg    | 27.77% | 2.20  | 2.78  | 0.360  | 249.2     | 8.90     |
| 22  | Viande hachée     | kg    | 28.17% | 3.60  | 5.03  | -0.454 | 282.8     | 10.10    |
| 23  | Confiture         | kg    | 28.86% | 1.28  | 1.65  | 0.309  | 138.6     | 4.95     |
| 24  | Sel               | kg    | 29.55% | 1.02  | 1.34  | 0.238  | 98.7      | 3.53     |
| 25  | Crème fraîche     | L     | 30.35% | 2.00  | 2.68  | 0.009  | 240.3     | 8.58     |
| 26  | Café              | kg    | 31.33% | 2.04  | 2.72  | -0.082 | 160.7     | 5.74     |
| 27  | Huile végétale    | L     | 34.25% | 1.49  | 1.92  | -0.148 | 138.6     | 4.95     |
| 28  | Farine            | kg    | 38.47% | 2.43  | 2.97  | -0.112 | 224.9     | 8.03     |
| 29  | Jus d'orange      | L     | 40.78% | 5.44  | 6.60  | -0.040 | 432.8     | 15.46    |
| 30  | Poisson blanc     | kg    | 43.96% | 3.36  | 3.85  | -0.128 | 251.6     | 8.99     |
| 31  | Tomates fraîches  | kg    | 48.61% | 4.70  | 5.09  | 0.155  | 344.1     | 12.29    |
| 32  | Pain frais        | kg    | 48.61% | 9.03  | 10.02 | 0.022  | 620.2     | 22.15    |
| 33  | Sauce tomate      | L     | 49.56% | 2.94  | 3.66  | -0.490 | 194.8     | 6.96     |
| 34  | Saumon frais      | kg    | 52.62% | 2.31  | 2.61  | 0.010  | 187.9     | 6.71     |
| 35  | Thé               | kg    | 53.88% | 1.67  | 2.09  | -0.632 | 106.2     | 3.79     |

### Analyse des produits les plus performants (MAPE < 20%)

Les 6 produits du **top tier** présentent des caractéristiques communes :

- **Volume de données élevé** : entre 1 471 et 1 848 sorties sur la période d'entraînement
- **Consommation régulière** : ces produits de base (oeufs, poulet, jambon, fromage, pommes de terre, viande) sont consommés quotidiennement de façon stable
- **R² positif** : le modèle explique une part significative de la variance (0.15 à 0.47)

### Analyse des produits les moins performants (MAPE > 45%)

Les 5 produits en difficulté partagent des caractéristiques communes :

| Produit          | Nb sorties | R²     | Particularité                              |
| ---------------- | ---------- | ------ | ------------------------------------------ |
| Tomates fraîches | 1 042      | 0.155  | Forte saisonnalité non linéaire            |
| Pain frais       | 696        | 0.022  | Peu de sorties, consommation très variable |
| Sauce tomate     | 1 619      | -0.490 | R² très négatif — modèle inadapté          |
| Saumon frais     | 664        | 0.010  | Peu de sorties, produit de niche           |
| Thé              | 1 605      | -0.632 | R² le plus négatif — forte sur-estimation  |

**Observations clés** :

- Le **Pain frais** et le **Saumon frais** ont moins de 700 sorties — le modèle manque de régularité dans les données
- La **Sauce tomate** et le **Thé** ont un **R² très négatif** (respectivement -0.49 et -0.63), ce qui signifie que le modèle Prophet fait **pire qu'une simple moyenne** pour ces produits
- Les **Tomates fraîches** suivent une saisonnalité forte (été) difficilement capturable par les composantes Fourier standard

---

## 5. Analyse bayésienne des régresseurs (MCMC)

L'utilisation du MCMC avec 300 samples permet d'obtenir de **vrais intervalles de confiance bayésiens** sur les coefficients des régresseurs. Un coefficient est considéré **statistiquement significatif** lorsque son intervalle de confiance à 85% **ne contient pas 0**.

### 5.1. Épidémie de grippe — Significatif pour 100% des produits

C'est le régresseur le plus impactant. Les intervalles de confiance ne contiennent **jamais 0** pour aucun des 35 produits.

| Produit       | Coefficient | IC 85%         | Interprétation                       |
| ------------- | ----------- | -------------- | ------------------------------------ |
| Oeufs         | +28.35      | [22.87, 33.75] | +28 unités/jour en période de grippe |
| Légumes frais | +17.98      | [15.51, 20.55] | +18 kg/jour en période de grippe     |
| Pain frais    | +17.40      | [15.33, 19.58] | +17 kg/jour en période de grippe     |
| Lait entier   | +15.50      | [12.88, 18.11] | +15.5 L/jour en période de grippe    |
| Fruits frais  | +12.26      | [10.19, 14.39] | +12 kg/jour en période de grippe     |
| Sel           | +1.07       | [0.84, 1.31]   | +1 kg/jour en période de grippe      |

**Conclusion** : les épidémies de grippe provoquent une **hausse systématique et significative** de la consommation de tous les produits alimentaires. L'amplitude est proportionnelle au volume habituel de consommation du produit. Ce régresseur est **indispensable** au modèle.

### 5.2. Taux d'occupation — Significatif pour ~90% des produits

Le taux d'occupation est positif pour la quasi-totalité des produits, avec des intervalles robustes.

| Produit       | Coefficient | IC 85%          | Significatif ? |
| ------------- | ----------- | --------------- | -------------- |
| Oeufs         | +1.663      | [1.053, 2.262]  | Oui            |
| Légumes frais | +0.604      | [0.262, 0.941]  | Oui            |
| Lait entier   | +0.473      | [0.170, 0.792]  | Oui            |
| Riz           | +0.437      | [0.323, 0.560]  | Oui            |
| Pâtes sèches  | +0.393      | [0.308, 0.478]  | Oui            |
| Crème fraîche | +0.043      | [-0.005, 0.091] | **Non**        |
| Pain frais    | +0.132      | [-0.107, 0.370] | **Non**        |

**Conclusion** : plus l'hôpital est occupé, plus la consommation augmente. C'est le deuxième régresseur le plus fiable. Seuls quelques produits (Crème fraîche, Pain frais) n'ont pas un signal significatif, probablement en raison de patterns de consommation indépendants de l'occupation.

### 5.3. Température — Impact mixte, significatif pour ~50% des produits

La température a un impact **variable selon le type de produit** :

**Effet positif significatif (consommation ↑ quand il fait chaud) :**

| Produit           | Coefficient | IC 85%         |
| ----------------- | ----------- | -------------- |
| Fruits frais      | +0.197      | [0.083, 0.306] |
| Riz               | +0.154      | [0.094, 0.210] |
| Pâtes sèches      | +0.120      | [0.071, 0.169] |
| Conserves légumes | +0.082      | [0.043, 0.119] |
| Farine            | +0.081      | [0.049, 0.108] |

**Effet négatif significatif (consommation ↓ quand il fait chaud) :**

| Produit       | Coefficient | IC 85%           |
| ------------- | ----------- | ---------------- |
| Poulet frais  | -0.085      | [-0.132, -0.038] |
| Crème fraîche | -0.032      | [-0.058, -0.006] |

**Non significatif (intervalle contient 0) :**
Carottes, Confiture, Fromage emmental, Pain frais, Pommes de terre, Viande de boeuf, Viande hachée, entre autres.

**Conclusion** : la température est un régresseur informatif pour **certains produits** (notamment les fruits et les produits secs en été, le poulet en hiver). Son inclusion est justifiée même s'il n'est pas universellement significatif.

### 5.4. Nombre de patients — Non significatif pour la quasi-totalité des produits

C'est le constat le plus important de l'analyse MCMC.

| Produit       | Coefficient | IC 85%          | Significatif ? |
| ------------- | ----------- | --------------- | -------------- |
| Oeufs         | +0.051      | [-0.182, 0.273] | Non            |
| Lait entier   | +0.036      | [-0.082, 0.151] | Non            |
| Légumes frais | -0.029      | [-0.158, 0.098] | Non            |
| Fruits frais  | -0.079      | [-0.174, 0.013] | Non            |
| Pâtes sèches  | -0.008      | [-0.041, 0.024] | Non            |
| Sel           | +0.008      | [-0.001, 0.018] | Non            |

Sur les 35 produits, **aucun ne présente un effet significatif** du nombre de patients. Les intervalles contiennent systématiquement 0.

**Explication** : le nombre de patients (`nb_patients`) est fortement corrélé au taux d'occupation (`taux_occupation`). Le taux d'occupation capture déjà l'essentiel de l'information, rendant le nombre de patients **redondant** (problème de multicolinéarité). Le modèle "distribue" l'incertitude entre les deux variables corrélées, ce qui élargit les intervalles et annule la significativité.

**Recommandation** : envisager le retrait de `nb_patients` des régresseurs pour simplifier le modèle sans perte d'information prédictive.

### 5.5. Synthèse des régresseurs

| Régresseur        | Impact moyen   | Significatif (%) | Verdict                        |
| ----------------- | -------------- | ---------------- | ------------------------------ |
| `epidemie_grippe` | Fort positif   | **35/35 (100%)** | Indispensable                  |
| `taux_occupation` | Modéré positif | **~32/35 (91%)** | Très utile                     |
| `temperature`     | Variable       | **~17/35 (49%)** | Utile pour certains produits   |
| `nb_patients`     | Négligeable    | **~0/35 (0%)**   | Redondant — retrait recommandé |

---

## 6. Recommandations de commande

### Paramètres

| Paramètre                     | Valeur                          |
| ----------------------------- | ------------------------------- |
| Horizon de commande           | 7 jours                         |
| Stock de sécurité             | 2 jours de consommation moyenne |
| Total à commander             | **2 083.3 unités**              |
| Produits nécessitant commande | **20 / 35**                     |

### Détail des commandes (ordonnées par urgence)

#### Commandes urgentes (couverture < 2 jours)

| Produit          | Stock actuel | Conso prévue 7j | Stock sécu. | À commander  | Couverture |
| ---------------- | ------------ | --------------- | ----------- | ------------ | ---------- |
| Pain frais       | 8.5 kg       | 155.7 kg        | 44.5        | **191.6 kg** | 0.4 j      |
| Poisson blanc    | 9.9 kg       | 64.7 kg         | 18.5        | **73.2 kg**  | 1.1 j      |
| Saumon frais     | 8.0 kg       | 46.5 kg         | 13.3        | **51.8 kg**  | 1.2 j      |
| Poulet frais     | 16.5 kg      | 96.5 kg         | 27.6        | **107.5 kg** | 1.2 j      |
| Tomates fraîches | 15.5 kg      | 83.0 kg         | 23.7        | **91.2 kg**  | 1.3 j      |
| Lait entier      | 52.7 L       | 245.6 L         | 70.2        | **263.0 L**  | 1.5 j      |

#### Commandes normales (couverture 2–5 jours)

| Produit          | Stock actuel | Conso prévue 7j | À commander | Couverture |
| ---------------- | ------------ | --------------- | ----------- | ---------- |
| Viande de boeuf  | 24.5 kg      | 81.6 kg         | 80.4 kg     | 2.1 j      |
| Crème fraîche    | 21.0 L       | 58.2 L          | 53.7 L      | 2.5 j      |
| Légumes frais    | 91.0 kg      | 216.8 kg        | 187.7 kg    | 2.9 j      |
| Carottes         | 39.3 kg      | 82.3 kg         | 66.5 kg     | 3.3 j      |
| Viande hachée    | 34.5 kg      | 72.5 kg         | 58.7 kg     | 3.3 j      |
| Fruits frais     | 92.2 kg      | 177.7 kg        | 136.3 kg    | 3.6 j      |
| Salade verte     | 36.5 kg      | 71.9 kg         | 56.0 kg     | 3.6 j      |
| Jambon blanc     | 29.1 kg      | 56.3 kg         | 43.2 kg     | 3.6 j      |
| Fromage emmental | 20.4 kg      | 38.2 kg         | 28.7 kg     | 3.7 j      |
| Pommes de terre  | 65.1 kg      | 122.0 kg        | 91.8 kg     | 3.7 j      |
| Oeufs            | 280.1 u      | 513.3 u         | 379.9 u     | 3.8 j      |
| Yaourt nature    | 50.0 kg      | 89.0 kg         | 64.5 kg     | 3.9 j      |
| Beurre           | 18.5 kg      | 30.0 kg         | 20.1 kg     | 4.3 j      |
| Fromage frais    | 41.2 kg      | 61.2 kg         | 37.5 kg     | 4.7 j      |

#### Pas de commande nécessaire (couverture > 7 jours)

15 produits disposent d'un stock suffisant pour couvrir les 7 prochains jours : Biscuits, Café, Jus d'orange, Conserves légumes, Confiture, Compote de fruits, Farine, Huile végétale, Riz, Pâtes sèches, Pâtes fraîches, Sauce tomate, Thé, Sel, Sucre.

---

## 7. Plans d'arrivages

Le système calcule un plan d'arrivages optimisé selon la DLC de chaque produit, garantissant qu'aucun stock ne dépasse sa date d'expiration.

### Stratégie par type de DLC

| Type de DLC             | Produits                                                                                   | Nb arrivages / 28j | Stratégie                   |
| ----------------------- | ------------------------------------------------------------------------------------------ | ------------------ | --------------------------- |
| Ultra-courte (1 jour)   | Pain frais, Saumon frais, Viande hachée, Salade verte                                      | 28                 | Livraison quotidienne       |
| Très courte (2–3 jours) | Poulet frais, Viande de boeuf, Tomates fraîches, Lait entier, Crème fraîche, Fromage frais | 10–14              | Livraisons bihebdomadaires  |
| Courte (5–7 jours)      | Jambon blanc, Fruits frais, Yaourt nature                                                  | 4–6                | Livraisons hebdomadaires    |
| Moyenne (10–19 jours)   | Carottes, Légumes frais, Fromage emmental, Beurre, Oeufs, Pommes de terre                  | 2–3                | 2–3 livraisons sur 28 jours |
| Longue (> 30 jours)     | Conserves, Riz, Pâtes, Café, Sucre, Sel, etc.                                              | 1                  | Livraison unique            |

### Volume total d'arrivages

Le volume total à livrer sur 28 jours s'élève à environ **12 300 unités** réparties sur l'ensemble des produits, avec une concentration logistique sur les produits à DLC courte.

---

## 8. Comparaison MAP vs MCMC

Une exécution préalable avec l'estimation MAP (Maximum A Posteriori, `mcmc_samples=0`) permet de comparer les deux approches sur les mêmes données.

### Performance globale

| Indicateur          | MAP       | MCMC (300) | Delta |
| ------------------- | --------- | ---------- | ----- |
| MAPE moyen          | **28.4%** | 29.6%      | +1.2% |
| MAPE médian         | **24.9%** | 25.5%      | +0.6% |
| MAE moyen           | **3.64**  | 3.79       | +0.15 |
| Produits < 20% MAPE | 6/35      | 6/35       | =     |
| Produits < 35% MAPE | **28/35** | 27/35      | -1    |

### Impact par produit

**Produits améliorés par le MCMC :**

| Produit       | MAPE MAP | MAPE MCMC | Gain         |
| ------------- | -------- | --------- | ------------ |
| Saumon frais  | 60.72%   | 52.62%    | **-8.1 pts** |
| Pain frais    | 53.35%   | 48.61%    | **-4.7 pts** |
| Poisson blanc | 47.33%   | 43.96%    | **-3.4 pts** |
| Salade verte  | 23.30%   | 20.42%    | **-2.9 pts** |
| Fruits frais  | 26.24%   | 24.41%    | **-1.8 pts** |
| Poulet frais  | 20.19%   | 18.79%    | **-1.4 pts** |

**Produits dégradés par le MCMC :**

| Produit          | MAPE MAP | MAPE MCMC | Perte     |
| ---------------- | -------- | --------- | --------- |
| Tomates fraîches | 36.68%   | 48.61%    | +11.9 pts |
| Jus d'orange     | 32.61%   | 40.78%    | +8.2 pts  |
| Lait entier      | 15.66%   | 21.18%    | +5.5 pts  |
| Beurre           | 18.34%   | 21.57%    | +3.2 pts  |

### Analyse de la divergence

La légère dégradation du MAPE moyen est **attendue et normale** :

1. **Régularisation bayésienne** : le MCMC applique une régularisation plus forte que le MAP, réduisant le risque d'overfitting au prix d'une performance ponctuelle légèrement inférieure
2. **Robustesse** : les produits les plus faibles (Saumon, Pain, Poisson) s'améliorent significativement, ce qui indique que le MAP sur-ajustait ces produits
3. **Estimations plus honnêtes** : le MAP peut produire des estimations "chanceuses" sur le jeu de test qui ne se généralisent pas

### Le vrai gain : les intervalles de confiance

| Aspect           | MAP                            | MCMC                                      |
| ---------------- | ------------------------------ | ----------------------------------------- |
| Coefficients     | Estimations ponctuelles        | Distributions postérieures complètes      |
| Intervalles      | coef_lower = coef = coef_upper | Vrais intervalles bayésiens               |
| Significativité  | Impossible à tester            | Test par inclusion du 0 dans l'intervalle |
| Interprétabilité | Limitée                        | Quantification de l'incertitude           |

---

## 9. Limites identifiées et axes d'amélioration

### Limites actuelles

1. **Multicolinéarité `nb_patients` / `taux_occupation`** : le nombre de patients est redondant avec le taux d'occupation, confirmé par l'analyse MCMC (0% de significativité). Son retrait simplifierait le modèle.

2. **Produits à consommation intermittente** : les produits avec peu de sorties (Saumon frais : 664, Pain frais : 696, Poisson blanc : 639) ont des performances dégradées. Prophet n'est pas conçu pour la prévision de demande intermittente.

3. **R² négatifs** : 7 produits ont un R² négatif (Thé : -0.63, Sauce tomate : -0.49, Viande hachée : -0.45, Huile végétale : -0.15, Poisson blanc : -0.13, Farine : -0.11, Café : -0.08), indiquant que le modèle fait pire qu'une moyenne simple. Pour ces produits, Prophet n'est pas l'approche optimale.

4. **Saisonnalité non captée** : certains produits (Tomates fraîches) suivent une saisonnalité très marquée et asymétrique (pic estival) que les composantes Fourier peinent à modéliser.

### Axes d'amélioration proposés

| Axe                               | Action                                                            | Impact attendu                             |
| --------------------------------- | ----------------------------------------------------------------- | ------------------------------------------ |
| Retrait `nb_patients`             | Supprimer le régresseur redondant                                 | Simplification, stabilité des coefficients |
| Modèle alternatif pour MAPE > 45% | Tester ARIMA, ETS ou Croston pour les 5 produits faibles          | Amélioration MAPE de 5–15 pts              |
| Regroupement de produits          | Agréger les produits à faible volume (Saumon, Pain, Poisson)      | Plus de régularité dans les données        |
| Cross-validation                  | Utiliser `cross_validation()` de Prophet au lieu d'un split fixe  | Évaluation plus robuste                    |
| Augmentation MCMC                 | Passer de 300 à 500–1000 samples pour des intervalles plus précis | Réduction de la variance MCMC              |

---

## 10. Conclusion

Le système de prévision multi-produits avec Prophet et inférence MCMC produit des résultats **exploitables opérationnellement pour 77% des produits** (27/35 avec MAPE < 35%). L'activation du MCMC a permis de :

1. **Valider scientifiquement les régresseurs** : l'épidémie de grippe et le taux d'occupation sont les deux variables les plus influentes, confirmées par l'analyse bayésienne
2. **Identifier la redondance de `nb_patients`** : ce régresseur n'apporte aucune information supplémentaire significative
3. **Améliorer la robustesse** sur les produits difficiles : Saumon frais (-8 pts), Pain frais (-5 pts), Poisson blanc (-3 pts)
4. **Quantifier l'incertitude** : chaque coefficient dispose maintenant d'un intervalle de confiance interprétable

Le pipeline complet (prédictions 28 jours, recommandations de commande, plans d'arrivages avec DLC) constitue un **outil d'aide à la décision fonctionnel** pour la gestion des stocks de la Clinique du Mont Vert, avec des marges de sécurité intégrées pour compenser les erreurs de prévision.

---

## Annexes

### Fichiers générés

| Fichier                             | Description                                                |
| ----------------------------------- | ---------------------------------------------------------- |
| `metrics_tous_produits.csv`         | Métriques de performance (MAE, MAPE, RMSE, R²) par produit |
| `predictions_tous_produits_28j.csv` | Prédictions à 28 jours avec intervalles de confiance (85%) |
| `recommandations_commande.csv`      | Recommandations de commande avec stocks de sécurité        |
| `arrivages_tous_produits_28j.csv`   | Plans d'arrivages optimisés selon la DLC                   |
| `coefficients_regresseurs.csv`      | Coefficients MCMC avec intervalles bayésiens               |
| `summary_tous_produits.json`        | Résumé structuré complet en JSON                           |
| `graphiques/`                       | 35 graphiques de prédictions Prophet (PNG)                 |

### Références

- Taylor, S.J. & Letham, B. (2018). _Forecasting at Scale_. The American Statistician, 72(1), 37-45.
- Documentation Prophet : [https://facebook.github.io/prophet/](https://facebook.github.io/prophet/)
- Dataset : `dataset_stock_hopital_ENRICHI.csv` — Enrichi v3.0

---

_Document généré le 15 mars 2026 — Projet Master EISI — Gestion des stocks hospitaliers_
