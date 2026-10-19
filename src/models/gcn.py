"""Modèle 3 : GCN sur le graphe de l'ARN (cours GNN : message passing).

Couche GCN (Kipf & Welling) :      X' = relu( D^-1/2 (A + I) D^-1/2  X W )
avec A = somme des relations (squelette + paires + BPPS au niveau "graph").

Option relational=True : R-GCN, un poids par type d'arête (cours GNN) :
    x'_v = relu( W_0 x_v + sum_r  Â_r X W_r )
"""
import keras
from keras import layers, ops

from .common import check_level, embed, model_inputs, output_head, used_inputs
from .graph import GCNNormalize, graph_adjacency


class GCNLayer(layers.Layer):
    def __init__(self, units, n_relations=1, **kwargs):
        super().__init__(**kwargs)
        self.self_loop = layers.Dense(units) if n_relations > 1 else None
        self.rel = [layers.Dense(units, use_bias=n_relations == 1) for _ in range(n_relations)]

    def call(self, x, A):
        h = self.self_loop(x) if self.self_loop is not None else 0.0
        for r, dense in enumerate(self.rel):
            h = h + ops.matmul(A[:, r], dense(x))  # message passing
        return ops.relu(h)


def build_model(n_feats, level="graph", relational=False, embed_dim=32, units=128, n_layers=4,
                dropout=0.2):
    check_level(level)
    inputs = model_inputs(n_feats)
    x = layers.Dense(units)(embed(inputs, embed_dim, level))

    A = graph_adjacency(x, inputs, level)                      # (B, R, L, L)
    if not relational:
        A = ops.sum(A, axis=1, keepdims=True)                  # une seule matrice
    A = GCNNormalize()(A)

    for _ in range(n_layers):
        h = GCNLayer(units, n_relations=A.shape[1])(x, A)
        x = layers.LayerNormalization()(x + layers.Dropout(dropout)(h))  # connexion résiduelle

    name = f"{'rgcn' if relational else 'gcn'}_{level}"
    return keras.Model(used_inputs(inputs, level, ("bpps", "pairs")), output_head(x), name=name)
