"""Entrées de la stratégie 2, lues dans data/processed/ (écrit par scripts/preprocess.py).

Mêmes données que experiments/from_scratch/train.py. Comme dans le reste du projet, un batch ne contient
qu'une seule longueur : un « groupe » = un dict de tableaux de même longueur L.
"""
import numpy as np

from src.preprocessing.data import inputs_of, load_processed, select

TL_KEYS = ["tokens", "feats", "bpps", "pairs"]   # onehot inutile : couche Embedding
PROCESSED = ("train", "test_107", "test_130")


def load_inputs(name):
    """Entrées du modèle pour 'train', 'test_107' ou 'test_130'."""
    return select(inputs_of(load_processed(name)), slice(None), TL_KEYS)


def sn_weight(signal_to_noise):
    """Poids optionnel par molécule (option --sn-weight)."""
    return (0.5 * np.log(np.clip(signal_to_noise, 0, None) + 1.1)).astype("float32")


def n_batches(groups, bs):
    return sum(-(-len(g["tokens"]) // bs) for g in groups)


def iterate_batches(groups, bs, shuffle=False, rng=None, device="cpu"):
    """Batches de longueur homogène convertis en tenseurs torch.
    Sans shuffle, l'ordre de sortie est celui des groupes puis des lignes."""
    import torch
    jobs = []
    for gi, g in enumerate(groups):
        n = len(g["tokens"])
        order = rng.permutation(n) if shuffle else np.arange(n)
        jobs += [(gi, order[i:i + bs]) for i in range(0, n, bs)]
    if shuffle:
        jobs = [jobs[j] for j in rng.permutation(len(jobs))]
    for gi, idx in jobs:
        yield {k: torch.from_numpy(np.ascontiguousarray(v[idx])).to(device)
               for k, v in groups[gi].items()}
