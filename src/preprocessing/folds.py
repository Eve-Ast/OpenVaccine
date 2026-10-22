"""Lire les partitions fixes, communes à tous les modèles.

Origine : KFold mélangé (seed 42) ; GroupKFold sur 100 clusters KMeans
(n_init=10, seed 42) de comptes de 3-mers. Les CSV versionnés font référence.
"""
import pandas as pd
from .. import config as C


def load_folds(name='kfold'):
    path = C.FOLDS_DIR / f'folds_{name}.csv'
    if not path.exists():
        raise FileNotFoundError(f'{path} absent : récupérer le fichier versionné dans le dépôt')
    folds = pd.read_csv(path)
    if folds.id.duplicated().any() or not folds.fold.isin(range(C.N_FOLDS)).all():
        raise ValueError(f'{path} : identifiants dupliqués ou numéros de fold incorrects')
    return folds
