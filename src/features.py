"""Feature engineering par base (Personne 1).

Ajouter ici les nouvelles features : elles arrivent automatiquement dans
l'entrée `feats` de tous les modèles.
"""
import numpy as np


def pair_index(structure):
    """Pour chaque position, l'index de la base appariée (-1 si non appariée)."""
    stack, pairs = [], [-1] * len(structure)
    for i, c in enumerate(structure):
        if c == "(":
            stack.append(i)
        elif c == ")":
            j = stack.pop()
            pairs[i], pairs[j] = j, i
    return pairs


def bpps_features(bpps):
    """(N, L, 3) : somme, max, et probabilité de ne pas être appariée."""
    s = bpps.sum(axis=2)
    m = bpps.max(axis=2)
    unpaired = 1.0 - s
    return np.stack([s, m, unpaired], axis=-1)


def pair_distance(df):
    """(N, L, 1) : distance |i - j| / L à la base appariée (0 si non appariée)."""
    out = []
    for s in df.structure:
        p = np.asarray(pair_index(s))
        d = np.where(p >= 0, np.abs(np.arange(len(s)) - p), 0) / len(s)
        out.append(d)
    return np.asarray(out, dtype="float32")[..., None]


def base_features(df, bpps):
    return np.concatenate([bpps_features(bpps), pair_distance(df)], axis=-1).astype("float32")
