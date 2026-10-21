"""Même BiGRU que la baseline, puis self-attention sans pooling."""
import keras
from keras import layers

from .common import check_level, embed, model_inputs, output_head, used_inputs
from .gru import recurrent_layers


def build_model(n_feats, level="annot", embed_dim=32, hidden=64, n_layers=2,
                dropout=0.2, num_heads=4):
    check_level(level)
    if level == "graph":
        raise ValueError("BiGRU + Attention accepte seq et annot")
    if 2 * hidden % num_heads:
        raise ValueError("2 * hidden doit être divisible par num_heads")
    inputs = model_inputs(n_feats)
    x = recurrent_layers(embed(inputs, embed_dim, level), hidden, n_layers, dropout)
    a = layers.MultiHeadAttention(num_heads=num_heads, key_dim=2 * hidden // num_heads,
                                  dropout=dropout)(x, x)
    x = layers.LayerNormalization()(x + layers.Dropout(dropout)(a))
    return keras.Model(used_inputs(inputs, level), output_head(x), name=f"attention_gru_{level}")
