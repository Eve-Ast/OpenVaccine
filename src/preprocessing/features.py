"""Feature engineering par base (Personne 1).

Ajouter ici les nouvelles features (et leur nom dans FEATURE_NAMES) : elles
arrivent automatiquement dans l'entrée `feats` de tous les modèles.
"""
import numpy as np

FEATURE_NAMES = ["bpp_sum", "bpp_max", "bpp_entropy", "pair_distance"]


def pair_index(structure):
    """Pour chaque position, l'index de la base appariée (-1 si non appariée)."""
    stack, pairs = [], [-1] * len(structure)
    for i, c in enumerate(structure):
        if c == "(":
            stack.append(i)
        elif c == ")":
            j = stack.pop()
            pairs[i], pairs[j] = j, i
    if stack:
        raise ValueError("structure non équilibrée")
    return pairs


def bpps_features(bpps):
    """(N, L, 3) : probabilité d'être appariée (somme), probabilité max, entropie.

    L'entropie inclut l'état « non appariée » (1 - somme) : elle est faible
    quand la base a un partenaire (ou une absence de partenaire) bien défini.
    """
    s = bpps.sum(axis=2)
    m = bpps.max(axis=2)
    p = np.concatenate([bpps, np.clip(1.0 - s, 0, 1)[..., None]], axis=2)
    entropy = -(p * np.log(p + 1e-12)).sum(axis=2)
    return np.stack([s, m, entropy], axis=-1)


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
