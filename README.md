# OpenVaccine : prédiction de la dégradation de l'ARNm

Projet IA2, sujet 4 : [Kaggle Stanford COVID Vaccine](https://www.kaggle.com/competitions/stanford-covid-vaccine)

Prédire, pour chaque base d'une molécule d'ARNm, sa réactivité et sa dégradation dans différentes conditions. C'est une régression multi-sorties par position, évaluée en **MCRMSE** sur 3 cibles : `reactivity`, `deg_Mg_pH10` et `deg_Mg_50C`.

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
├── train.json         (ou train.json.zip, les deux marchent)
├── test.json
├── sample_submission.csv
└── bpps/              (6034 fichiers .npy)
```

## Structure

```
├── src/
│   ├── config.py        constantes partagées (cibles, chemins, seed)
│   ├── data.py          chargement et tenseurs d'entrée (tokens, feats, bpps, pairs)
│   ├── features.py      feature engineering                       [P1]
│   ├── folds.py         découpage KFold / GroupKFold              [P1]
│   ├── metrics.py       MCRMSE (métrique) et loss masquée         [P1]
│   ├── submission.py    format Kaggle
│   └── models/
│       ├── gru.py          Modèle 1  : BiGRU baseline             [P1]
│       ├── conv_gru.py     Modèle 2  : Conv1D + BiGRU             [P2]
│       ├── transformer.py  Modèle 3a : Transformer + biais BPPS   [P2]
│       └── gnn.py          Modèle 3b : R-GCN sur le graphe d'ARN  [P3]
├── folds/               folds fixés, communs à toute l'équipe (ne pas régénérer)
├── baseline.py          prédicteur naïf (moyenne par position)    [P1]
├── train.py             entraînement en validation croisée (tous les modèles)
├── ensemble.py          ensembling pondéré des modèles            [P3]
├── notebooks/           EDA, analyses, figures pour le rapport
└── report/              rapport et slides
```

## Utilisation

```bash
python baseline.py                                   # référence naïve
python train.py --model gru                          # Modèle 1
python train.py --model conv_gru                     # Modèle 2
python train.py --model transformer                  # Modèle 3a
python train.py --model gnn                          # Modèle 3b
python ensemble.py --runs gru conv_gru transformer gnn

# Test rapide : un seul fold, quelques époques
python train.py --model gnn --only-fold 0 --epochs 3 --name test

# Hyperparamètres et optimiseurs (étude d'optimisation)
python train.py --model conv_gru --optimizer nesterov --lr 1e-2 --batch-size 32 --name conv_gru_nesterov
python train.py --model transformer --params '{"use_bpps": false}' --name transformer_no_bpps
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
- Les fichiers communs (`src/config.py`, `data.py`, `metrics.py`, `folds.py`) appartiennent à P1. Toute modification se fait par PR, prévenez le groupe.
- Pour ajouter une feature : modifiez `src/features.py`. Elle arrive automatiquement dans l'entrée `feats` de tous les modèles.
- Ne jamais committer `data/` ni `outputs/`. Partagez les scores dans le tableau de résultats du rapport.
- `folds/*.csv` sont figés : ne pas les régénérer, sinon les scores ne sont plus comparables.
