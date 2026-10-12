"""Prédicteur naïf (Personne 1) : moyenne de chaque cible par position.

Point de référence : tout modèle doit faire mieux que ça.
    python baseline.py
"""
import json

import numpy as np

from src import config as C
from src.data import build_targets, load_train
from src.folds import load_folds
from src.metrics import mcrmse

train = load_train().merge(load_folds("kfold"), on="id")
Y = build_targets(train)
clean = (train.SN_filter == 1).to_numpy()

oof = np.zeros_like(Y)
for k in range(C.N_FOLDS):
    is_val = (train.fold == k).to_numpy()
    oof[is_val] = Y[~is_val & clean].mean(axis=0)

score = mcrmse(Y[clean], oof[clean])
print(f"Baseline moyenne par position : MCRMSE = {score:.4f}")
out = C.OUTPUT_DIR / "mean_baseline"
out.mkdir(parents=True, exist_ok=True)
np.save(out / "oof.npy", oof)
(out / "scores.json").write_text(json.dumps({"cv_mcrmse": score}, indent=2))
