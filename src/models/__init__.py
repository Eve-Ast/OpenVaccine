from . import conv_gru, gnn, gru, transformer

MODELS = {
    "gru": gru.build_model,               # Modèle 1  - Personne 1
    "conv_gru": conv_gru.build_model,     # Modèle 2  - Personne 2
    "transformer": transformer.build_model,  # Modèle 3a - Personne 2
    "gnn": gnn.build_model,               # Modèle 3b - Personne 3
}
