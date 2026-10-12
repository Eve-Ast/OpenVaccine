"""Constantes partagées par toute l'équipe (chemins, cibles, seed)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
BPPS_DIR = DATA_DIR / "bpps"
FOLDS_DIR = ROOT / "folds"
OUTPUT_DIR = ROOT / "outputs"

# Les 5 cibles, dans l'ordre du sample_submission
TARGETS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
# Les 3 cibles évaluées par Kaggle (MCRMSE)
SCORED_TARGETS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
SCORED_IDX = [TARGETS.index(t) for t in SCORED_TARGETS]  # [0, 1, 3]

SEQ_SCORED = 68  # seules les 68 premières bases sont annotées dans train

# Vocabulaires des 3 entrées texte. Un seul vocabulaire global avec des
# offsets pour pouvoir utiliser une seule couche Embedding.
SEQ_VOCAB = "ACGU"
STRUCT_VOCAB = "()."
LOOP_VOCAB = "SMIBHEX"
VOCAB_SIZE = len(SEQ_VOCAB) + len(STRUCT_VOCAB) + len(LOOP_VOCAB)  # 14

SEED = 42
N_FOLDS = 5
