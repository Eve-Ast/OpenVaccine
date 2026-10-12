"""Modèle 1 (Personne 1) : baseline BiGRU 2 couches."""
import keras
from keras import layers

from .common import embed, model_inputs, output_head


def build_model(n_feats, embed_dim=32, hidden=128, n_layers=2, dropout=0.3, use_feats=False):
    inputs = model_inputs(n_feats)
    x = embed(inputs, embed_dim, use_feats)
    for _ in range(n_layers):
        x = layers.Bidirectional(layers.GRU(hidden, return_sequences=True, dropout=dropout))(x)
    used = {k: inputs[k] for k in (["tokens", "feats"] if use_feats else ["tokens"])}
    return keras.Model(used, output_head(x), name="bigru")
