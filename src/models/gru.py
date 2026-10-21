"""BiGRU à deux couches, régression par nucléotide, poids initialisés aléatoirement."""
import keras
from keras import layers

from .common import check_level, embed, model_inputs, output_head, used_inputs


def recurrent_layers(x, hidden=64, n_layers=2, dropout=0.2):
    """Même bloc récurrent pour la baseline et ses variantes."""
    for _ in range(n_layers):
        x = layers.Bidirectional(layers.GRU(hidden, return_sequences=True))(x)
        x = layers.Dropout(dropout)(x)
    return x


def build_model(n_feats, level="annot", embed_dim=32, hidden=64, n_layers=2,
                dropout=0.2):
    check_level(level)
    if level == "graph":
        raise ValueError("BiGRU accepte les niveaux seq et annot")
    inputs = model_inputs(n_feats)
    x = embed(inputs, embed_dim, level)
    x = recurrent_layers(x, hidden, n_layers, dropout)
    return keras.Model(used_inputs(inputs, level), output_head(x), name=f"gru_{level}")
