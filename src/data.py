"""Chargement des données et construction des tenseurs d'entrée.

Toutes les molécules d'un même batch ont la même longueur (107 en train,
107 ou 130 en test), donc pas besoin de padding ni de masque.
"""
import numpy as np
import pandas as pd

from . import config as C
from .features import base_features, pair_index


def _read(name):
    path = C.DATA_DIR / f"{name}.json"
    if not path.exists():
        path = C.DATA_DIR / f"{name}.json.zip"
    return pd.read_json(path, lines=True)


def load_train():
    return _read("train")


def load_test():
    return _read("test")


def encode_tokens(df):
    """(N, L, 3) entiers : [base, structure, type de boucle] avec offsets."""
    seq_map = {c: i for i, c in enumerate(C.SEQ_VOCAB)}
    off = len(C.SEQ_VOCAB)
    struct_map = {c: off + i for i, c in enumerate(C.STRUCT_VOCAB)}
    off += len(C.STRUCT_VOCAB)
    loop_map = {c: off + i for i, c in enumerate(C.LOOP_VOCAB)}

    tokens = [
        np.stack([
            [seq_map[c] for c in row.sequence],
            [struct_map[c] for c in row.structure],
            [loop_map[c] for c in row.predicted_loop_type],
        ], axis=-1)
        for row in df.itertuples()
    ]
    return np.asarray(tokens, dtype="int32")


def load_bpps(ids):
    """(N, L, L) : matrices de probabilité d'appariement."""
    return np.asarray([np.load(C.BPPS_DIR / f"{i}.npy") for i in ids], dtype="float32")


def build_inputs(df):
    """Dictionnaire d'entrées commun à tous les modèles.

    tokens : (N, L, 3)  int    -> Embedding
    feats  : (N, L, F)  float  -> features numériques par base (features.py)
    bpps   : (N, L, L)  float  -> matrice BPPS (Transformer / GNN)
    pairs  : (N, L)     int    -> index de la base appariée, -1 sinon (GNN)
    """
    assert df.seq_length.nunique() == 1, "grouper les molécules par longueur"
    bpps = load_bpps(df.id)
    return {
        "tokens": encode_tokens(df),
        "feats": base_features(df, bpps),
        "bpps": bpps,
        "pairs": np.asarray([pair_index(s) for s in df.structure], dtype="int32"),
    }


def build_targets(df):
    """(N, 68, 5) dans l'ordre de config.TARGETS."""
    return np.stack([np.asarray(df[t].tolist(), dtype="float32") for t in C.TARGETS], axis=-1)


def select(inputs, idx, keys=None):
    """Sous-ensemble de lignes (idx) et d'entrées (keys = entrées du modèle)."""
    return {k: inputs[k][idx] for k in (keys or inputs)}
