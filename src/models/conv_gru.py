"""Motifs locaux par Conv1D, puis même BiGRU que la baseline."""
import keras
from keras import layers

from .common import check_level, embed, model_inputs, output_head, used_inputs
from .gru import recurrent_layers


def build_model(n_feats, level="annot", embed_dim=32, hidden=64, n_layers=2,
                dropout=0.2, filters=64, kernel_size=5):
    check_level(level)
    if level == "graph":
        raise ValueError("Conv1D + BiGRU accepte seq et annot")
    inputs = model_inputs(n_feats)
    x = embed(inputs, embed_dim, level)
    x = layers.Conv1D(filters, kernel_size, padding="same", activation="relu")(x)
    x = recurrent_layers(x, hidden, n_layers, dropout)
    return keras.Model(used_inputs(inputs, level), output_head(x), name=f"conv_gru_{level}")
