"""Découpage train / validation (Personne 1).

Appelé par preprocess.py : crée folds/folds_kfold.csv et folds/folds_group.csv
s'ils n'existent pas. Ces fichiers sont committés et ne sont jamais
régénérés : toute l'équipe utilise exactement les mêmes folds.

- kfold : KFold classique.
- group : GroupKFold sur des clusters de séquences similaires (KMeans sur les
  comptes de 3-mers), pour que des séquences proches ne soient pas à la fois
  en train et en validation (cf. cours PLM : fuite entre homologues).
"""
from itertools import product

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold, KFold

from .. import config as C

KMERS = ["".join(p) for p in product(C.SEQ_VOCAB, repeat=3)]


def kmer_counts(seq, k=3):
    counts = dict.fromkeys(KMERS, 0)
    for i in range(len(seq) - k + 1):
        counts[seq[i : i + k]] += 1
    return list(counts.values())


def make_folds(df, strategy="kfold", n_clusters=100):
    fold = np.zeros(len(df), dtype=int)
    if strategy == "kfold":
        splits = KFold(C.N_FOLDS, shuffle=True, random_state=C.SEED).split(df)
    elif strategy == "group":
        X = np.asarray([kmer_counts(s) for s in df.sequence], dtype=float)
        groups = KMeans(n_clusters, random_state=C.SEED, n_init=10).fit_predict(X)
        splits = GroupKFold(C.N_FOLDS).split(df, groups=groups)
    else:
        raise ValueError(strategy)
    for k, (_, val_idx) in enumerate(splits):
        fold[val_idx] = k
    return pd.DataFrame({"id": df.id, "fold": fold})


def load_folds(name="kfold"):
    return pd.read_csv(C.FOLDS_DIR / f"folds_{name}.csv")


def ensure_folds(train):
    """Crée les fichiers de folds manquants ; ne touche jamais aux fichiers existants."""
    C.FOLDS_DIR.mkdir(exist_ok=True)
    for strategy in ["kfold", "group"]:
        path = C.FOLDS_DIR / f"folds_{strategy}.csv"
        if not path.exists():
            make_folds(train, strategy).to_csv(path, index=False)
            print(f"folds créés : {path}")
