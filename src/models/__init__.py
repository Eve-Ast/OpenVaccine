from . import attention_gru, conv_gru, gat, gcn, gru, mlp, transformer

MODELS = {
    "gru": gru.build_model,                # baseline architecturale
    "conv_gru": conv_gru.build_model,
    "attention_gru": attention_gru.build_model,
    "mlp": mlp.build_model,                  # référence auxiliaire, chaque base seule
    "transformer": transformer.build_model,  # Modèle 2 : self-attention (+ biais BPPS)
    "gcn": gcn.build_model,                  # Modèle 3 : GCN (option relational=True : R-GCN)
    "gat": gat.build_model,                  # attention sur graphe
}
