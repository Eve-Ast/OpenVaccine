"""Prédicteur naïf : moyenne de chaque cible par position.

Point de référence (aucun apprentissage) : tout modèle doit faire mieux.
    python -m experiments.from_scratch.train --model mean
"""
import numpy as np


class MeanBaseline:
    def __init__(self, Y):
        """Y : cibles d'entraînement (N, 68, 5)."""
        self.per_position = Y.mean(axis=0)        # (68, 5)
        self.overall = Y.mean(axis=(0, 1))       # (5,) pour les positions > 68 du test

    def predict(self, inputs, **_):
        n, L = inputs["pairs"].shape
        out = np.tile(self.overall, (n, L, 1)).astype("float32")
        k = min(L, len(self.per_position))
        out[:, :k] = self.per_position[:k]
        return out
