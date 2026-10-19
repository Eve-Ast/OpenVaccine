"""Prétraitement des données (Personne 1).

    python preprocess.py

1. Vérifie les données brutes et écrit un rapport (outputs/preprocessing_report.json) :
   valeurs manquantes, vocabulaires, longueurs, structures, BPPS, cibles, bruit.
2. Crée les folds s'ils n'existent pas encore (folds/, communs à l'équipe).
3. Encode les entrées et écrit les données prétraitées dans data/processed/ :
   train.npz, test_107.npz, test_130.npz (le test contient deux longueurs).

Contenu de chaque .npz :
    ids                (N,)
    tokens             (N, L, 3)   base / structure / boucle -> Embedding
    onehot             (N, L, 14)  même information en one-hot
    feats              (N, L, F)   features numériques (src/features.py)
    bpps               (N, L, L)   matrices de probabilité d'appariement
    pairs              (N, L)      index de la base appariée (-1 sinon)
  train seulement :
    targets            (N, 68, 5)  cibles (ordre config.TARGETS)
    errors             (N, 68, 5)  incertitudes expérimentales
    signal_to_noise    (N,)
    sn_filter          (N,)        1 = molécule fiable
    fold_kfold, fold_group (N,)    folds communs à l'équipe

Les molécules bruitées (SN_filter = 0) sont gardées dans le fichier : le choix
de les exclure se fait à l'entraînement (train.py, option --all-samples).
Les cibles ne sont ni modifiées ni tronquées : la métrique Kaggle porte sur
les valeurs brutes.
"""
import json

import numpy as np

from src import config as C
from src.preprocessing.data import build_errors, build_inputs, build_targets, load_test, load_train
from src.preprocessing.features import FEATURE_NAMES, pair_index
from src.preprocessing.folds import ensure_folds, load_folds


def describe(a):
    a = a.ravel()
    return {"min": round(float(a.min()), 3), "p1": round(float(np.percentile(a, 1)), 3),
            "median": round(float(np.median(a)), 3), "p99": round(float(np.percentile(a, 99)), 3),
            "max": round(float(a.max()), 3), "pct_negative": round(100 * float((a < 0).mean()), 2)}


def check(train, test):
    report = {}
    report["n_molecules"] = {"train": len(train), "test": len(test)}
    report["missing_values"] = {"train": int(train.isna().sum().sum()), "test": int(test.isna().sum().sum())}
    report["seq_length"] = {"train": train.seq_length.value_counts().to_dict(),
                            "test": test.seq_length.value_counts().to_dict()}
    report["seq_scored"] = {"train": train.seq_scored.value_counts().to_dict(),
                            "test": test.seq_scored.value_counts().to_dict()}

    both = [train, test]
    for col, vocab in [("sequence", C.SEQ_VOCAB), ("structure", C.STRUCT_VOCAB),
                       ("predicted_loop_type", C.LOOP_VOCAB)]:
        seen = set("".join(s for df in both for s in df[col]))
        assert seen <= set(vocab), f"{col} : caractères inconnus {seen - set(vocab)}"
        assert all(len(s) == L for df in both for s, L in zip(df[col], df.seq_length)), f"{col} : longueur"
    for df in both:
        for s in df.structure:
            pair_index(s)  # lève une erreur si la structure n'est pas équilibrée
    report["vocab_and_structure"] = "OK"

    targets = build_targets(train)
    errors = build_errors(train)
    clean = (train.SN_filter == 1).to_numpy()
    report["sn_filter"] = {"fiables (SN_filter = 1)": int(clean.sum()), "bruitées": int((~clean).sum()),
                           "signal_to_noise": describe(train.signal_to_noise.to_numpy())}
    report["targets_all"] = {t: describe(targets[..., i]) for i, t in enumerate(C.TARGETS)}
    report["targets_sn_filter_1"] = {t: describe(targets[clean][..., i]) for i, t in enumerate(C.TARGETS)}
    report["errors_all"] = {e: describe(errors[..., i]) for i, e in enumerate(C.ERROR_COLS)}
    report["errors_sn_filter_1"] = {e: describe(errors[clean][..., i]) for i, e in enumerate(C.ERROR_COLS)}
    return report, targets, errors


def check_bpps(inputs, name, report):
    b = inputs["bpps"]
    sym = float(np.abs(b - b.transpose(0, 2, 1)).max())
    rows = b.sum(axis=2)
    report.setdefault("bpps", {})[name] = {
        "shape": list(b.shape), "min": round(float(b.min()), 4), "max": round(float(b.max()), 4),
        "max_asymmetry": round(sym, 6), "max_row_sum": round(float(rows.max()), 4)}
    assert b.min() >= 0 and b.max() <= 1 + 1e-6, f"{name} : BPPS hors de [0, 1]"


def main():
    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    C.OUTPUT_DIR.mkdir(exist_ok=True)
    train, test = load_train(), load_test()

    report, targets, errors = check(train, test)

    # ---- train
    ensure_folds(train)
    folds = {name: load_folds(name).set_index("id").fold for name in ["kfold", "group"]}
    X = build_inputs(train)
    check_bpps(X, "train", report)
    np.savez_compressed(
        C.PROCESSED_DIR / "train.npz", ids=train.id.to_numpy().astype(str), **X,
        targets=targets, errors=errors,
        signal_to_noise=train.signal_to_noise.to_numpy("float32"),
        sn_filter=train.SN_filter.to_numpy("int8"),
        fold_kfold=folds["kfold"].loc[train.id].to_numpy("int8"),
        fold_group=folds["group"].loc[train.id].to_numpy("int8"))

    # ---- test, un fichier par longueur
    for L, g in test.groupby("seq_length"):
        Xg = build_inputs(g)
        check_bpps(Xg, f"test_{L}", report)
        np.savez_compressed(C.PROCESSED_DIR / f"test_{L}.npz", ids=g.id.to_numpy().astype(str), **Xg)

    report["features"] = FEATURE_NAMES
    report["feature_stats_train"] = {n: describe(X["feats"][..., i]) for i, n in enumerate(FEATURE_NAMES)}
    (C.OUTPUT_DIR / "preprocessing_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print(json.dumps({k: report[k] for k in ["n_molecules", "seq_length", "sn_filter", "bpps"]},
                     indent=2, ensure_ascii=False))
    print("\nCibles (SN_filter = 1) :")
    for t, d in report["targets_sn_filter_1"].items():
        print(f"  {t:12s} {d}")
    print(f"\n-> {C.PROCESSED_DIR}  |  rapport : {C.OUTPUT_DIR / 'preprocessing_report.json'}")


if __name__ == "__main__":
    main()
