from . import gat, gcn, mlp, transformer

MODELS = {
    "mlp": mlp.build_model,                  # Modèle 1 : baseline, chaque base seule
    "transformer": transformer.build_model,  # Modèle 2 : self-attention (+ biais BPPS)
    "gcn": gcn.build_model,                  # Modèle 3 : GCN (option relational=True : R-GCN)
    "gat": gat.build_model,                  # Modèle 4 (optionnel) : attention sur graphe
}
