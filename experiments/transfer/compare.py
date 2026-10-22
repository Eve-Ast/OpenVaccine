"""Comparaison scientifique des modèles (Personne C), même interface qu'scripts/ensemble.py.

    python -m experiments.transfer.compare --runs tl tl_scratch gru gnn --folds group

Lit outputs/<run>/oof.npy et écrit dans results/transfer/ :
main_table.csv, bootstrap.csv, loop_errors_<run>.csv.
Évaluation sur les molécules fiables (SN_filter = 1) prédites par tous les modèles.
"""
import argparse

import numpy as np
import pandas as pd

from src import config as C
from src.evaluation.metrics import mcrmse
from src.preprocessing.data import load_processed

ALL_IDX = list(range(len(C.TARGETS)))
LOOP_OFFSET = len(C.SEQ_VOCAB) + len(C.STRUCT_VOCAB)


def mcrmse_from_se(se, cols):
    """se : (N, 5) somme des erreurs² par molécule -> même formule que metrics.mcrmse."""
    return float(np.sqrt(se.sum(0) / (se.shape[0] * C.SEQ_SCORED))[cols].mean())


def paired_bootstrap(oof_a, oof_b, Y, cols, B=2000, seed=C.SEED):
    """Rééchantillonnage des molécules (unité indépendante), pas des positions."""
    se_a, se_b = ((oof_a - Y) ** 2).sum(1), ((oof_b - Y) ** 2).sum(1)
    rng, N = np.random.default_rng(seed), len(Y)
    d = np.empty(B)
    for i in range(B):
        idx = rng.integers(0, N, N)
        d[i] = mcrmse_from_se(se_a[idx], cols) - mcrmse_from_se(se_b[idx], cols)
    obs = mcrmse_from_se(se_a, cols) - mcrmse_from_se(se_b, cols)
    return obs, np.percentile(d, [2.5, 97.5]), float((d >= 0).mean())


def error_by_loop_type(P, Y, loops):
    rows = []
    for c in C.LOOP_VOCAB:
        m = loops == c
        if m.any():
            rows.append({"loop": c, "n_pos": int(m.sum()),
                         **{t: float(np.sqrt(((P[..., k] - Y[..., k]) ** 2)[m].mean()))
                            for k, t in enumerate(C.TARGETS)}})
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True, help="dossiers dans outputs/")
    p.add_argument("--folds", default="kfold", choices=["kfold", "group"],
                   help="folds utilisés par les runs (pour les scores par fold)")
    a = p.parse_args()
    out = C.ROOT / "results" / "transfer"
    out.mkdir(parents=True, exist_ok=True)

    train = load_processed("train")
    Y, fold = train["targets"], train[f"fold_{a.folds}"]
    loop_idx = train["tokens"][:, : C.SEQ_SCORED, 2] - LOOP_OFFSET
    loops = np.asarray(list(C.LOOP_VOCAB))[loop_idx]
    oofs = {r: np.load(C.OUTPUT_DIR / r / "oof.npy") for r in a.runs}
    clean = (train["sn_filter"] == 1) & np.all([~np.isnan(o).any(axis=(1, 2)) for o in oofs.values()], axis=0)
    print(f"{clean.sum()} molécules utilisées pour l'évaluation")
    Yc, fc, lc = Y[clean], fold[clean], loops[clean]
    oofs = {r: o[clean] for r, o in oofs.items()}

    rows = []
    for r, P in oofs.items():
        per_fold = [mcrmse(Yc[fc == f], P[fc == f]) for f in np.unique(fc)]
        rows.append({"run": r,
                     "MCRMSE (officielle, 3 cibles)": mcrmse(Yc, P),
                     "par fold (moy ± std)": f"{np.mean(per_fold):.4f} ± {np.std(per_fold):.4f}",
                     "MCRMSE 5 cibles": mcrmse(Yc, P, cols=ALL_IDX),
                     **{t: mcrmse(Yc, P, cols=[k]) for k, t in enumerate(C.TARGETS)}})
    table = pd.DataFrame(rows).sort_values("MCRMSE (officielle, 3 cibles)")
    table.to_csv(out / "main_table.csv", index=False)
    print("\n=== Tableau principal ===")
    print(table.to_markdown(index=False, floatfmt=".4f"))

    names, pairs = list(oofs), []
    print("\n=== Bootstrap apparié, MCRMSE officielle (Δ = A − B ; négatif = A meilleur) ===")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            obs, ci, pv = paired_bootstrap(oofs[names[i]], oofs[names[j]], Yc, C.SCORED_IDX)
            pairs.append({"A": names[i], "B": names[j], "delta": obs,
                          "ic95_bas": ci[0], "ic95_haut": ci[1], "P(delta>=0)": pv})
            print(f"{names[i]} vs {names[j]} : Δ={obs:+.4f}  IC95 [{ci[0]:+.4f}, {ci[1]:+.4f}]  P(Δ≥0)={pv:.3f}")
    pd.DataFrame(pairs).to_csv(out / "bootstrap.csv", index=False)

    for r, P in oofs.items():
        df = error_by_loop_type(P, Yc, lc)
        df.to_csv(out / f"loop_errors_{r}.csv", index=False)
        print(f"\n=== RMSE par type de boucle : {r} ===")
        print(df.to_markdown(index=False, floatfmt=".4f"))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
