"""Quatre caractéristiques par nucléotide : somme, maximum, entropie BPPS et distance."""
import numpy as np

FEATURE_NAMES = ["bpp_sum", "bpp_max", "bpp_entropy", "pair_distance"]


def pair_index(structure):
    """Pour chaque position, l'index de la base appariée (-1 si non appariée)."""
    stack, pairs = [], [-1] * len(structure)
    for i, c in enumerate(structure):
        if c == "(":
            stack.append(i)
        elif c == ")":
            if not stack:
                raise ValueError("structure non équilibrée")
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


def base_features(pairs, bpps):
    """Concaténer les trois statistiques BPPS et la distance au partenaire."""
    length = pairs.shape[1]
    distance = np.where(pairs >= 0, np.abs(np.arange(length) - pairs), 0) / length
    distance = distance.astype('float32')[..., None]
    return np.concatenate([bpps_features(bpps), distance], axis=-1).astype('float32')
