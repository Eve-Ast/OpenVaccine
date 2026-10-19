"""Modèle 1 : MLP (baseline).

Le même réseau dense est appliqué à chaque base indépendamment : aucune
information sur les bases voisines, ni sur les interactions. C'est la
référence minimale (cours Introduction ML/DL : perceptron multicouche).
"""
import keras
from keras import layers

from .common import check_level, embed, model_inputs, output_head, used_inputs


def build_model(n_feats, level="annot", embed_dim=32, hidden=(256, 128), dropout=0.3):
    check_level(level)
    if level == "graph":
        raise ValueError("le MLP traite chaque base seule : level = 'seq' ou 'annot'")
    inputs = model_inputs(n_feats)
    x = embed(inputs, embed_dim, level)
    for units in hidden:
        x = layers.Dense(units, activation="relu")(x)
        x = layers.Dropout(dropout)(x)
    return keras.Model(used_inputs(inputs, level), output_head(x), name=f"mlp_{level}")
