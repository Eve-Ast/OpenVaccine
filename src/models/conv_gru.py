"""Modèle 2 (Personne 2) : Conv1D + BiGRU profond avec Spatial Dropout."""
import keras
from keras import layers

from .common import embed, model_inputs, output_head


def build_model(n_feats, embed_dim=32, filters=128, kernel_sizes=(3, 5, 7), hidden=128,
                n_gru=3, dropout=0.3, spatial_dropout=0.2):
    inputs = model_inputs(n_feats)
    x = embed(inputs, embed_dim)
    x = layers.SpatialDropout1D(spatial_dropout)(x)

    # Convolutions 1D : motifs locaux (voisinage de la base)
    for k in kernel_sizes:
        x = layers.Conv1D(filters, k, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.SpatialDropout1D(spatial_dropout)(x)

    # BiGRU : dépendances longues le long de la chaîne
    for _ in range(n_gru):
        x = layers.Bidirectional(layers.GRU(hidden, return_sequences=True, dropout=dropout))(x)

    used = {k: inputs[k] for k in ["tokens", "feats"]}
    return keras.Model(used, output_head(x), name="conv_bigru")
