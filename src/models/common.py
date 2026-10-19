"""Briques partagées par tous les modèles.

Niveau d'information structurale (`level`), commun à tous les modèles, pour
répondre à la question : les interactions structurales améliorent-elles la
prédiction par rapport aux modèles séquentiels ?

  "seq"   : séquence de nucléotides seule
  "annot" : + annotations structurales par base (structure, type de boucle,
            features BPPS par base) -- sans les interactions entre bases
  "graph" : + interactions entre bases (matrice BPPS / paires), utilisées par
            l'attention (Transformer) ou comme arêtes (GCN, GAT)
"""
import keras
from keras import layers

from .. import config as C

LEVELS = ("seq", "annot", "graph")


def model_inputs(n_feats):
    """Entrées Keras ; L = None pour accepter 107 et 130."""
    return {
        "tokens": keras.Input((None, 3), dtype="int32", name="tokens"),
        "feats": keras.Input((None, n_feats), name="feats"),
        "bpps": keras.Input((None, None), name="bpps"),
        "pairs": keras.Input((None,), dtype="int32", name="pairs"),
    }


def check_level(level):
    if level not in LEVELS:
        raise ValueError(f"level doit être dans {LEVELS}, reçu {level!r}")


def embed(inputs, embed_dim, level="annot"):
    """Embedding des tokens (+ features numériques) -> (B, L, D).

    level = "seq" : seulement le token de la base ; sinon les 3 tokens et les
    features par base.
    """
    check_level(level)
    tokens = inputs["tokens"][:, :, :1] if level == "seq" else inputs["tokens"]
    n_tok = 1 if level == "seq" else 3
    x = layers.Embedding(C.VOCAB_SIZE, embed_dim)(tokens)  # (B, L, n_tok, E)
    x = layers.Reshape((-1, n_tok * embed_dim))(x)
    if level != "seq":
        x = layers.Concatenate()([x, inputs["feats"]])
    return x


def used_inputs(inputs, level, graph_keys=("bpps",)):
    """Entrées réellement utilisées par le modèle, selon le niveau."""
    keys = ["tokens"] + ([] if level == "seq" else ["feats"]) + (list(graph_keys) if level == "graph" else [])
    return {k: inputs[k] for k in keys}


def output_head(x):
    return layers.Dense(len(C.TARGETS), name="targets")(x)
