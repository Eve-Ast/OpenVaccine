# OpenVaccine : prédiction de la dégradation de l'ARNm

Projet IA2, sujet 4 : [Kaggle Stanford COVID Vaccine](https://www.kaggle.com/competitions/stanford-covid-vaccine)

Prédire, pour chaque base d'une molécule d'ARNm, sa réactivité et sa dégradation dans différentes conditions. C'est une régression multi-sorties par position, évaluée en **MCRMSE** sur 3 cibles : `reactivity`, `deg_Mg_pH10` et `deg_Mg_50C`.

**Question de recherche :** la prise en compte des interactions structurales entre nucléotides améliore-t-elle la prédiction de la dégradation de l'ARN par rapport aux modèles séquentiels ?

Pour y répondre, chaque modèle est entraîné à plusieurs **niveaux d'information structurale** (`--level`) :

| Niveau | Entrées | MLP | Transformer | GCN / GAT |
|---|---|---|---|---|
| `seq` | séquence seule | ✓ | ✓ | ✓ (graphe = squelette) |
| `annot` | + structure, type de boucle, features BPPS par base | ✓ | ✓ | ✓ (graphe = squelette) |
| `graph` | + interactions entre bases (BPPS, paires) | — | ✓ (biais d'attention) | ✓ (arêtes paires + BPPS) |

L'apport des interactions structurales se lit dans l'écart `annot` → `graph` à architecture égale.

**Répartition des tâches, du rapport et de la soutenance : [docs/REPARTITION.md](docs/REPARTITION.md)**

## Installation

```bash
git clone <url-du-repo>
cd OpenVaccine
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Données (non versionnées)

Les données ne sont pas versionnées. Téléchargez-les depuis Kaggle et placez-les dans `data/` :

```bash
kaggle competitions download -c stanford-covid-vaccine
unzip stanford-covid-vaccine.zip -d data/
```

```
data/
├── train.json         (train.json.zip est aussi accepté)
├── test.json
├── sample_submission.csv
└── bpps/              (6034 fichiers .npy)
```

### Prétraitement (à lancer une fois)

```bash
python preprocess.py
```

Vérifie les données et écrit `outputs/preprocessing_report.json`. Le rapport contient les valeurs manquantes, les vocabulaires, la validité des structures, les BPPS, les statistiques des cibles et le bruit. Le script crée ensuite les données prétraitées dans `data/processed/`, utilisées par `train.py` et `ensemble.py` :

| Donnée brute | Prétraitement | Clé dans les `.npz` |
|---|---|---|
| `sequence`, `structure`, `predicted_loop_type` | Tokens (pour l'Embedding) et one-hot (14 dims) | `tokens`, `onehot` |
| `bpps` | Matrice complète + features par base : somme, max, entropie | `bpps`, `feats` |
| `structure` | Index de la base appariée, distance à la paire | `pairs`, `feats` |
| `reactivity`, `deg_*` | Vérifiées, non modifiées ; loss masquée sur les 68 premières bases | `targets`, `errors` |
| `signal_to_noise`, `SN_filter` | Conservés ; les molécules bruitées sont exclues à l'entraînement et toujours en validation | `signal_to_noise`, `sn_filter` |
| Split train/validation | 5 folds fixés (KFold et GroupKFold) | `fold_kfold`, `fold_group` |

Le test contient deux longueurs (107 et 130), d'où deux fichiers : `test_107.npz` et `test_130.npz`.

## Architecture

Le projet suit trois étapes. Chaque étape a **un seul script** à la racine, et son code dans **un dossier dédié** de `src/` :

| Étape | Script | Code | Entrée → sortie |
|---|---|---|---|
| 1. Prétraitement | `preprocess.py` | `src/preprocessing/` | `data/*.json`, `data/bpps/` → `data/processed/*.npz` + rapport |
| 2. Modélisation | `train.py` | `src/models/` | `data/processed/` → `outputs/<run>/` |
| 3. Évaluation | `ensemble.py` | `src/evaluation/` | `outputs/<run>/` → `outputs/ensemble/` |

```
OpenVaccine/
├── preprocess.py            ÉTAPE 1 : vérification + données prétraitées
├── train.py                 ÉTAPE 2 : entraînement en validation croisée (tous les modèles)
├── ensemble.py              ÉTAPE 3 : moyenne pondérée des modèles
│
├── src/
│   ├── config.py            constantes partagées (chemins, cibles, vocabulaires, seed)
│   │
│   ├── preprocessing/       ── étape 1
│   │   ├── data.py          lecture JSON/BPPS, encodage (tokens, one-hot), lecture des .npz
│   │   ├── features.py      features par base (BPPS, distance à la paire)
│   │   └── folds.py         split train/validation (KFold, GroupKFold)
│   │
│   ├── models/              ── étape 2 (un fichier par modèle)
│   │   ├── common.py        entrées, embedding, niveaux seq / annot / graph
│   │   ├── graph.py         graphe de l'ARN (squelette, paires, BPPS) pour GCN et GAT
│   │   ├── mean.py          référence naïve : moyenne par position
│   │   ├── mlp.py           Modèle 1 : MLP
│   │   ├── transformer.py   Modèle 2 : Transformer Encoder (+ biais BPPS)
│   │   ├── gcn.py           Modèle 3 : GCN (option R-GCN)
│   │   └── gat.py           Modèle 4 (optionnel) : GAT
│   │
│   └── evaluation/          ── étape 3
│       ├── metrics.py       MCRMSE (métrique) et loss masquée
│       └── submission.py    format de soumission Kaggle
│
├── data/                    (non versionné) données brutes + data/processed/
├── folds/                   folds fixés, communs à l'équipe (créés une seule fois)
├── outputs/                 (non versionné) un dossier par run : scores, courbes, prédictions
├── notebooks/               EDA, analyses, figures du rapport
├── report/                  rapport et slides
└── docs/                    répartition des tâches
```

## Utilisation

```bash
python preprocess.py                                 # une fois (et après chaque modif. de features.py)
python train.py --model mean                         # référence naïve
python train.py --model mlp --level annot            # Modèle 1   -> outputs/mlp_annot/
python train.py --model transformer --level annot    # Modèle 2, sans interactions
python train.py --model transformer --level graph    # Modèle 2, attention biaisée par la BPPS
python train.py --model gcn --level annot            # Modèle 3, graphe = squelette seul
python train.py --model gcn --level graph            # Modèle 3, + paires et BPPS
python train.py --model gat --level graph            # Modèle 4 (optionnel)
python ensemble.py --runs mlp_annot transformer_graph gcn_graph

# Test rapide : un seul fold, quelques époques
python train.py --model gat --level graph --only-fold 0 --epochs 3 --name test

# Hyperparamètres et optimiseurs (étude d'optimisation)
python train.py --model transformer --level graph --optimizer nesterov --lr 1e-2 --batch-size 32 --name transformer_nesterov
python train.py --model gcn --level graph --params '{"relational": true}' --name rgcn_graph
```

Chaque run écrit dans `outputs/<name>/` :

| Fichier | Contenu | Utilisé pour |
|---|---|---|
| `scores.json` | MCRMSE CV, RMSE par cible, paramètres | tableau de résultats |
| `history_fold*.csv` | courbes de loss train / val | figures du rapport |
| `oof.npy` | prédictions out-of-fold | ensembling, analyse d'erreurs |
| `submission.csv` | prédictions test | Kaggle |

Toutes les entrées ont une longueur `L` variable : 107 en train, 107 ou 130 en test. Chaque modèle reçoit un dictionnaire :

- `tokens` (L, 3) : base, structure, type de boucle
- `feats` (L, F) : features numériques
- `bpps` (L, L) : matrice BPPS
- `pairs` (L,) : index de la base appariée

Chaque modèle prédit (L, 5). La loss n'utilise que les 68 premières positions.

## Workflow Git

- Une branche par personne : `p1-data`, `p2-sequence`, `p3-graph`. On fusionne dans `main` via Pull Request.
- Les fichiers communs (`src/config.py`, `src/preprocessing/`, `src/evaluation/`) appartiennent à P1. Toute modification se fait par PR, prévenez le groupe.
- Pour ajouter une feature : modifiez `src/preprocessing/features.py` (fonction + nom dans `FEATURE_NAMES`), puis relancez `python preprocess.py`. Elle arrive automatiquement dans l'entrée `feats` de tous les modèles.
- Ne jamais committer `data/` ni `outputs/`. Partagez les scores dans le tableau de résultats du rapport.
- `folds/*.csv` sont figés : ne pas les régénérer, sinon les scores ne sont plus comparables.
