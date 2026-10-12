"""Modèle 3b (Personne 3) : GNN relationnel (R-GCN) sur le graphe de l'ARN.

Nœuds = bases. Trois relations (une matrice d'adjacence par relation) :
  1. squelette covalent : i <-> i+1
  2. paires de bases de `structure` : i <-> pairs[i]
  3. BPPS : arête pondérée par la probabilité d'appariement

Couche R-GCN (cours GNN) :
  x'_v = relu( W_0 x_v + sum_r  D_r^-1/2 (A_r + I) D_r^-1/2  X W_r )
Les matrices sont denses : L <= 130, donc pas besoin de PyTorch Geometric.
"""
import keras
from keras import layers, ops

from .common import embed, model_inputs, output_head


class RnaGraph(layers.Layer):
    """Construit les adjacences normalisées (B, R, L, L) à partir de pairs et bpps."""

    def call(self, pairs, bpps):
        L = ops.shape(pairs)[1]
        idx = ops.arange(L)
        backbone = ops.cast(ops.abs(idx[:, None] - idx[None, :]) == 1, "float32")
        backbone = ops.broadcast_to(backbone[None], ops.shape(bpps))
        structure = ops.cast(pairs[:, :, None] == idx[None, None, :], "float32")
        A = ops.stack([backbone, structure, bpps], axis=1)  # (B, R, L, L)

        A = A + ops.eye(L)[None, None]  # self-loops : Â = A + I
        d_inv_sqrt = 1.0 / ops.sqrt(ops.sum(A, axis=-1))
        return d_inv_sqrt[..., :, None] * A * d_inv_sqrt[..., None, :]


class RGCNLayer(layers.Layer):
    def __init__(self, units, n_relations=3, **kwargs):
        super().__init__(**kwargs)
        self.self_loop = layers.Dense(units)
        self.rel = [layers.Dense(units, use_bias=False) for _ in range(n_relations)]

    def call(self, x, A):
        h = self.self_loop(x)
        for r, dense in enumerate(self.rel):
            h = h + ops.matmul(A[:, r], dense(x))  # message passing
        return ops.relu(h)


def build_model(n_feats, embed_dim=32, units=128, n_layers=4, dropout=0.2):
    inputs = model_inputs(n_feats)
    x = layers.Dense(units)(embed(inputs, embed_dim))
    A = RnaGraph()(inputs["pairs"], inputs["bpps"])

    for _ in range(n_layers):
        h = RGCNLayer(units)(x, A)
        x = layers.LayerNormalization()(x + layers.Dropout(dropout)(h))  # connexion résiduelle

    used = {k: inputs[k] for k in ["tokens", "feats", "bpps", "pairs"]}
    return keras.Model(used, output_head(x), name="rgcn")
