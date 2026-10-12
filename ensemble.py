"""Ensembling (Personne 3) : moyenne pondérée des modèles entraînés.

Les poids sont optimisés sur les prédictions OOF (pas sur le test !).
    python ensemble.py --runs gru conv_gru transformer gnn
"""
import argparse
import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src import config as C
from src.data import build_targets, load_train
from src.metrics import mcrmse

p = argparse.ArgumentParser()
p.add_argument("--runs", nargs="+", required=True, help="dossiers dans outputs/")
p.add_argument("--name", default="ensemble")
args = p.parse_args()

train = load_train()
Y = build_targets(train)
oofs = [np.load(C.OUTPUT_DIR / r / "oof.npy") for r in args.runs]
# molécules fiables et prédites par tous les modèles (tous les folds entraînés)
clean = (train.SN_filter == 1).to_numpy() & np.all([~np.isnan(o).any(axis=(1, 2)) for o in oofs], axis=0)
print(f"{clean.sum()} molécules utilisées pour l'évaluation")
oofs = [o[clean] for o in oofs]
for r, o in zip(args.runs, oofs):
    print(f"{r:20s} MCRMSE = {mcrmse(Y[clean], o):.4f}")


def blend(w, arrays):
    w = np.abs(w) / np.abs(w).sum()
    return sum(wi * a for wi, a in zip(w, arrays))


n = len(args.runs)
res = minimize(lambda w: mcrmse(Y[clean], blend(w, oofs)), np.ones(n) / n, method="Nelder-Mead")
weights = np.abs(res.x) / np.abs(res.x).sum()
score = mcrmse(Y[clean], blend(weights, oofs))
print("poids :", dict(zip(args.runs, weights.round(3))))
print(f"Ensemble MCRMSE = {score:.4f}")

subs = [pd.read_csv(C.OUTPUT_DIR / r / "submission.csv") for r in args.runs]
sub = subs[0].copy()
sub[C.TARGETS] = blend(weights, [s[C.TARGETS].to_numpy() for s in subs])
out = C.OUTPUT_DIR / args.name
out.mkdir(parents=True, exist_ok=True)
sub.to_csv(out / "submission.csv", index=False)
(out / "scores.json").write_text(json.dumps(
    {"cv_mcrmse": score, "weights": dict(zip(args.runs, weights.tolist()))}, indent=2))
print(f"-> {out}")
