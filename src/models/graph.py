"""Graphe de l'ARN commun au GCN et au GAT.

Nœuds = bases. Relations (une matrice d'adjacence (B, L, L) chacune) :
  1. squelette covalent : i <-> i+1                       (tous les niveaux)
  2. paires de bases de `structure` : i <-> pairs[i]      (level = "graph")
  3. BPPS : arête pondérée par la probabilité d'appariement (level = "graph")

Aux niveaux "seq" et "annot", le graphe n'a que le squelette : le GNN se
comporte alors comme un modèle séquentiel (voisins i-1, i+1). La différence
de score avec le niveau "graph" mesure l'apport des interactions structurales.
Les matrices sont denses : L <= 130, donc pas besoin de PyTorch Geometric.
"""
from keras import layers, ops


class RnaGraph(layers.Layer):
    """Renvoie les adjacences (B, R, L, L), R = 1 (squelette) ou 3 (squelette, paires, BPPS)."""

    def call(self, x, pairs=None, bpps=None):
        B, L = ops.shape(x)[0], ops.shape(x)[1]
        idx = ops.arange(L)
        backbone = ops.cast(ops.abs(idx[:, None] - idx[None, :]) == 1, "float32")
        backbone = ops.broadcast_to(backbone[None], (B, L, L))
        if pairs is None:
            return backbone[:, None]
        structure = ops.cast(pairs[:, :, None] == idx[None, None, :], "float32")
        return ops.stack([backbone, structure, bpps], axis=1)


class GCNNormalize(layers.Layer):
    """Normalisation symétrique du cours GNN : D^-1/2 (A + I) D^-1/2 (sur les 2 derniers axes)."""

    def call(self, A):
        A = A + ops.eye(ops.shape(A)[-1])
        d_inv_sqrt = 1.0 / ops.sqrt(ops.sum(A, axis=-1))
        return d_inv_sqrt[..., :, None] * A * d_inv_sqrt[..., None, :]


class NeighbourMask(layers.Layer):
    """Masque binaire (B, L, L) des voisins (toutes relations confondues) + la base elle-même."""

    def call(self, A):
        return ops.minimum(ops.sum(A, axis=1) + ops.eye(ops.shape(A)[-1]), 1.0)


def graph_adjacency(x, inputs, level):
    """Adjacences brutes (B, R, L, L) selon le niveau d'information structurale."""
    if level == "graph":
        return RnaGraph()(x, inputs["pairs"], inputs["bpps"])
    return RnaGraph()(x)
