"""Chargement des données et construction des tenseurs d'entrée.

Toutes les molécules d'un même batch ont la même longueur (107 en train,
107 ou 130 en test), donc pas besoin de padding ni de masque.
"""
import numpy as np
import pandas as pd

from .. import config as C
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
    onehot : (N, L, 14) float  -> one-hot des 3 entrées texte
    feats  : (N, L, F)  float  -> features numériques par base (features.py)
    bpps   : (N, L, L)  float  -> matrice BPPS (Transformer / GNN)
    pairs  : (N, L)     int    -> index de la base appariée, -1 sinon (GNN)
    """
    assert df.seq_length.nunique() == 1, "grouper les molécules par longueur"
    bpps = load_bpps(df.id)
    tokens = encode_tokens(df)
    return {
        "tokens": tokens,
        "onehot": one_hot(tokens),
        "feats": base_features(df, bpps),
        "bpps": bpps,
        "pairs": np.asarray([pair_index(s) for s in df.structure], dtype="int32"),
    }


def one_hot(tokens):
    """(N, L, 14) : one-hot de la base (4), de la structure (3) et de la boucle (7).

    Alternative à l'Embedding pour les modèles sans couche Embedding
    (baselines tabulaires, MLP).
    """
    return np.eye(C.VOCAB_SIZE, dtype="float32")[tokens].sum(axis=2)


def build_targets(df):
    """(N, 68, 5) dans l'ordre de config.TARGETS."""
    return np.stack([np.asarray(df[t].tolist(), dtype="float32") for t in C.TARGETS], axis=-1)


def build_errors(df):
    """(N, 68, 5) : incertitude expérimentale de chaque cible."""
    return np.stack([np.asarray(df[e].tolist(), dtype="float32") for e in C.ERROR_COLS], axis=-1)


# ------------------------------------------------------------------
# Données prétraitées (écrites par preprocess.py)
# ------------------------------------------------------------------
def load_processed(name):
    """name = 'train', 'test_107' ou 'test_130' -> dict de tableaux numpy."""
    path = C.PROCESSED_DIR / f"{name}.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} absent : lancer d'abord `python preprocess.py`")
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


INPUT_KEYS = ["tokens", "onehot", "feats", "bpps", "pairs"]


def inputs_of(processed):
    """Garde seulement les entrées des modèles d'un dictionnaire prétraité."""
    return {k: processed[k] for k in INPUT_KEYS}


def select(inputs, idx, keys=None):
    """Sous-ensemble de lignes (idx) et d'entrées (keys = entrées du modèle)."""
    return {k: inputs[k][idx] for k in (keys or inputs)}
