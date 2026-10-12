# Répartition des tâches, du rapport et de la soutenance

| | Nom | Branche Git | Rôle |
|---|---|---|---|
| **P1** | _______ | `p1-data` | Données, validation, features, Modèle 1 |
| **P2** | _______ | `p2-sequence` | Modèles séquentiels (Modèle 2, Transformer) et optimisation |
| **P3** | _______ | `p3-graph` | GNN, analyse d'erreurs, ensembling, résultats finaux |

Règle d'or : **mêmes folds, même MCRMSE, même protocole pour tous les modèles**. Sinon les scores ne sont pas comparables (cours PLM : *« comparer exige le même readout, le même split et le même budget »*).

---

## 1. Travail technique

### 👤 P1 : Données, Validation et Modèle 1

| Tâche | Fichiers |
|---|---|
| Définition du problème, MCRMSE et loss masquée (68 premières bases) | `src/metrics.py` |
| **EDA** : valeurs manquantes, types, longueurs (107 / 130), distributions des cibles (valeurs négatives, bruit, extrêmes), colonnes `*_error`, `signal_to_noise` / `SN_filter`, réactivité selon la base, le type de boucle et la position, corrélations entre cibles | `notebooks/01_eda.ipynb` |
| **Validation** : KFold vs GroupKFold (clusters de séquences similaires) | `src/folds.py`, `folds/` |
| **Feature engineering** : stats BPPS, distance à la base appariée, nouvelles features issues de l'analyse d'erreurs de P3 | `src/features.py` |
| Prédicteur naïf, puis **Modèle 1** : BiGRU 2 couches, avec et sans features | `baseline.py`, `src/models/gru.py` |

**Cours utilisés :**
- *Introduction ML/DL* : protocole ML, K-fold, métriques de régression, comparaison à un prédicteur naïf
- *PLM* : embeddings, séparation des homologues pour éviter la fuite d'information

### 👤 P2 : Modèles séquentiels et Optimisation

| Tâche | Fichiers |
|---|---|
| **Modèle 2** : Conv1D, puis BiGRU profond, avec Spatial Dropout | `src/models/conv_gru.py` |
| **Modèle 3a** : Transformer (Embedding, encodage positionnel sinusoïdal, Multi-Head Attention, LayerNorm). La BPPS est ajoutée en biais dans les scores d'attention. | `src/models/transformer.py` |
| **Ablation** : Transformer avec et sans BPPS (`--params '{"use_bpps": false}'`) | |
| **Étude d'optimisation** : SGD, Momentum, Nesterov, Adagrad, RMSProp et Adam ; ratio learning rate / batch ; LR range test ; dropout ; early stopping | `notebooks/02_optimisation.ipynb` |
| *(Bonus)* Pré-entraînement du Transformer par masked language modeling | |

**Cours utilisés :**
- *Optimisation pour le DL* : optimiseurs, ratio lr/batch, régularisation
- *Transformers* et *Transformers2* : attention, encodage positionnel, TP de prédiction de structure secondaire
- *Introduction ML/DL* : convolution 1D sur les séquences

### 👤 P3 : GNN, Analyse d'erreurs, Ensembling

| Tâche | Fichiers |
|---|---|
| **Modèle 3b** : l'ARN comme graphe. Nœuds = bases ; 3 relations = squelette (i, i+1), paires de `structure`, BPPS. Progression : GCN, puis R-GCN, puis (bonus) GAT. | `src/models/gnn.py` |
| **EDA secondaire** : analyse des erreurs sur les OOF (molécules, bases, boucles, positions les plus mal prédites). Les idées de features sont transmises à P1. | `notebooks/03_erreurs.ipynb` |
| **Ensembling** : poids optimisés sur les OOF | `ensemble.py` |
| **Soumission Kaggle finale** (late submission, scores public et privé) et tableau comparatif | |

**Cours utilisés :**
- *GNN* : message passing, GCN, R-GCN, GAT, Graph Transformer
- *Transformers* : un Transformer est un GNN sur un graphe complet

---

## 2. Rapport : qui écrit quoi

Chaque section indique son auteur (*« Section rédigée par … »*) et cite le cours utilisé.

| # | Section | Auteur | Contenu attendu |
|---|---|---|---|
| — | Résumé | **P3** (relu par tous) | ½ page : problème, approche, meilleur score |
| 1 | Introduction et contexte | **P1** | ARNm, instabilité, stockage entre -20 °C et -80 °C, objectif du projet |
| 2 | Données et EDA | **P1** | Description des colonnes, figures EDA, bruit et `SN_filter` |
| 3 | Formulation du problème et métrique | **P1** | Entrée (L, F), sortie (L, 5), MCRMSE, loss masquée |
| 4 | Stratégie de validation | **P1** | KFold vs GroupKFold, justification |
| 5 | Feature engineering | **P1** | Encodages, statistiques BPPS, apport de chaque feature |
| 6 | Modèle 1 : BiGRU baseline | **P1** | Architecture, résultat vs prédicteur naïf |
| 7 | Modèle 2 : Conv1D + BiGRU | **P2** | Architecture, rôle de la convolution et du Spatial Dropout |
| 8 | Modèle 3a : Transformer + BPPS | **P2** | Attention biaisée par la BPPS, ablation avec et sans BPPS |
| 9 | Étude d'optimisation | **P2** | Courbes des optimiseurs, ratio lr/batch, régularisation |
| 10 | Modèle 3b : GNN (R-GCN) | **P3** | Construction du graphe, équations, résultats |
| 11 | Analyse d'erreurs | **P3** | Où les modèles se trompent, et pourquoi |
| 12 | Ensembling et résultats finaux | **P3** (chacun remplit sa ligne) | Tableau comparatif de tous les modèles, score Kaggle |
| 13 | Conclusion et perspectives | **P3** (idées de tous) | Limites, pistes (pré-entraînement, données 130 bases, etc.) |
| — | Bibliographie | Tous | Cours IA2 et articles cités |

**Tableau de résultats commun** (section 12, chacun ajoute ses lignes) :

| Modèle | Auteur | CV MCRMSE | reactivity | deg_Mg_pH10 | deg_Mg_50C | Kaggle privé |
|---|---|---|---|---|---|---|
| Moyenne par position | P1 | 0.4257 | | | | |
| BiGRU (Modèle 1) | P1 | | | | | |
| Conv1D + BiGRU (Modèle 2) | P2 | | | | | |
| Transformer sans BPPS | P2 | | | | | |
| Transformer + BPPS (Modèle 3a) | P2 | | | | | |
| R-GCN (Modèle 3b) | P3 | | | | | |
| **Ensemble** | P3 | | | | | |

Les valeurs se trouvent dans `outputs/<run>/scores.json`.

---

## 3. Soutenance : qui présente quoi

Environ 6 minutes par personne, suivies des questions. Chacun répond aux questions sur sa partie.

### P1 : « Le problème et les données » (≈ 6 min)
1. Contexte : pourquoi l'ARNm se dégrade, et l'enjeu de la chaîne du froid
2. Les données : une molécule (séquence, structure, boucles) et une heatmap de BPPS
3. EDA : 2 ou 3 figures clés (distribution des cibles, réactivité par type de boucle, bruit)
4. Formulation : tenseur (L, F) en entrée, (L, 5) en sortie, MCRMSE, loss masquée
5. Validation : pourquoi GroupKFold
6. Baseline : prédicteur naïf vs BiGRU, pour montrer que le pipeline marche

### P2 : « Modèles séquentiels et optimisation » (≈ 6 min)
1. Modèle 2 : schéma Conv1D + BiGRU, avec le rôle de chaque bloc
2. Transformer : schéma de l'attention, et comment la BPPS biaise l'attention
3. Ablation avec et sans BPPS : l'information structurelle aide-t-elle ?
4. Étude d'optimisation : courbes de loss des optimiseurs, ratio lr/batch
5. Ce qu'on retient (le meilleur réglage)

### P3 : « Graphe, ensemble et résultats » (≈ 6 min)
1. L'ARN comme graphe : schéma des 3 types d'arêtes
2. R-GCN : l'équation de message passing (cours GNN)
3. Analyse d'erreurs : où les modèles échouent
4. Ensemble : poids obtenus et gain
5. Tableau final et score Kaggle
6. Conclusion et perspectives

---

## 4. Planning (à adapter à la date de rendu)

| Semaine | P1 | P2 | P3 |
|---|---|---|---|
| 1 | EDA, validation, features de base, BiGRU | Prise en main, Modèle 2 | Prise en main, GCN |
| 2 | Nouvelles features (retours de P3) | Transformer et ablation BPPS | R-GCN, analyse d'erreurs |
| 3 | Rédaction des sections 1 à 6 | Étude d'optimisation, rédaction des sections 7 à 9 | Ensemble, Kaggle, rédaction des sections 10 à 13 |
| 4 | Slides et répétition (tous ensemble) | | |
