"""Briques partagées par tous les modèles."""
import keras
from keras import layers

from .. import config as C


def model_inputs(n_feats):
    """Entrées Keras ; L = None pour accepter 107 et 130."""
    return {
        "tokens": keras.Input((None, 3), dtype="int32", name="tokens"),
        "feats": keras.Input((None, n_feats), name="feats"),
        "bpps": keras.Input((None, None), name="bpps"),
        "pairs": keras.Input((None,), dtype="int32", name="pairs"),
    }


def embed(inputs, embed_dim, use_feats=True):
    """Embedding des 3 tokens (concaténés) + features numériques -> (B, L, D)."""
    x = layers.Embedding(C.VOCAB_SIZE, embed_dim)(inputs["tokens"])  # (B, L, 3, E)
    x = layers.Reshape((-1, 3 * embed_dim))(x)
    if use_feats:
        x = layers.Concatenate()([x, inputs["feats"]])
    return x


def output_head(x):
    return layers.Dense(len(C.TARGETS), name="targets")(x)
