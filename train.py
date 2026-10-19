"""Entraînement en validation croisée, commun à tous les modèles.

Exemples :
    python train.py --model mean                       # référence naïve
    python train.py --model mlp --level annot
    python train.py --model transformer --level graph
    python train.py --model gcn --level annot          # GCN sur le squelette seul
    python train.py --model transformer --optimizer rmsprop --lr 3e-4 --batch-size 32
    python train.py --model gat --only-fold 0 --epochs 5   # test rapide

Sorties dans outputs/<name>/ :
    oof.npy            prédictions out-of-fold (2400, 68, 5) -> ensembling
    submission.csv     moyenne des 5 folds sur le test       -> Kaggle
    scores.json        MCRMSE CV, RMSE par cible, paramètres -> rapport
    history_fold*.csv  courbes de loss train/val             -> rapport
"""
import argparse
import json

import keras
import numpy as np

from src import config as C
from src.evaluation.metrics import mcrmse, mcrmse_loss
from src.evaluation.submission import to_submission
from src.models import MODELS
from src.models.common import LEVELS
from src.models.mean import MeanBaseline
from src.preprocessing.data import inputs_of, load_processed, select

# Optimiseurs vus dans le cours « Méthodes d'optimisation pour le DL »
OPTIMIZERS = {
    "sgd": lambda lr: keras.optimizers.SGD(lr),
    "momentum": lambda lr: keras.optimizers.SGD(lr, momentum=0.9),
    "nesterov": lambda lr: keras.optimizers.SGD(lr, momentum=0.9, nesterov=True),
    "adagrad": lambda lr: keras.optimizers.Adagrad(lr),
    "rmsprop": lambda lr: keras.optimizers.RMSprop(lr),
    "adam": lambda lr: keras.optimizers.Adam(lr),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["mean", *MODELS])
    p.add_argument("--name", help="nom du dossier de sortie (défaut : nom du modèle)")
    p.add_argument("--level", choices=LEVELS,
                   help="information structurale : seq | annot | graph (défaut : celui du modèle)")
    p.add_argument("--params", default="{}", help="hyperparamètres du modèle en JSON")
    p.add_argument("--folds", default="kfold", choices=["kfold", "group"])
    p.add_argument("--only-fold", type=int, help="n'entraîner qu'un fold (tests rapides)")
    p.add_argument("--optimizer", default="adam", choices=OPTIMIZERS)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--all-samples", action="store_true",
                   help="entraîner aussi sur les molécules bruitées (SN_filter = 0)")
    return p.parse_args()


def main():
    args = parse_args()
    params = json.loads(args.params)
    if args.level:
        params["level"] = args.level
    name = args.name or "_".join([args.model] + ([args.level] if args.level else []))
    out_dir = C.OUTPUT_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    keras.utils.set_random_seed(C.SEED)

    # données prétraitées par preprocess.py
    train = load_processed("train")
    X, Y = inputs_of(train), train["targets"]
    fold = train[f"fold_{args.folds}"]
    clean = train["sn_filter"] == 1
    n_feats = X["feats"].shape[-1]
    test_groups = [load_processed(f"test_{L}") for L in (107, 130)]

    folds = [args.only_fold] if args.only_fold is not None else range(C.N_FOLDS)
    oof = np.full_like(Y, np.nan)
    test_preds = [np.zeros((*g["pairs"].shape, len(C.TARGETS))) for g in test_groups]
    fold_scores = {}

    for k in folds:
        print(f"\n===== {name} | fold {k} =====")
        is_val = fold == k
        tr = ~is_val & (True if args.all_samples else clean)
        va = is_val & clean  # validation toujours sur les molécules fiables

        if args.model == "mean":   # référence naïve, pas d'apprentissage
            model, keys = MeanBaseline(Y[tr]), ["pairs"]
        else:
            model = MODELS[args.model](n_feats, **params)
            keys = [t.name for t in model.inputs]
            model.compile(optimizer=OPTIMIZERS[args.optimizer](args.lr), loss=mcrmse_loss)
            if k == folds[0]:
                model.summary()
            model.fit(
                select(X, tr, keys), Y[tr],
                validation_data=(select(X, va, keys), Y[va]),
                batch_size=args.batch_size, epochs=args.epochs, verbose=2,
                callbacks=[
                    keras.callbacks.EarlyStopping(patience=args.patience, restore_best_weights=True),
                    keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=max(1, args.patience // 2)),
                    keras.callbacks.CSVLogger(out_dir / f"history_fold{k}.csv"),
                ],
            )

        pred = model.predict(select(X, is_val, keys), batch_size=256, verbose=0)
        oof[is_val] = pred[:, : C.SEQ_SCORED]
        fold_scores[k] = mcrmse(Y[va], oof[va])
        print(f"fold {k} MCRMSE = {fold_scores[k]:.4f}")

        for i, g in enumerate(test_groups):
            test_preds[i] += model.predict(select(g, slice(None), keys), batch_size=256, verbose=0) / len(folds)

    done = np.isin(fold, list(folds)) & clean
    per_target = np.sqrt(np.nanmean((Y[done] - oof[done]) ** 2, axis=(0, 1)))
    scores = {
        "cv_mcrmse": mcrmse(Y[done], oof[done]),
        "fold_mcrmse": fold_scores,
        "rmse_per_target": dict(zip(C.TARGETS, per_target.round(4).tolist())),
        "args": vars(args),
    }
    print(json.dumps(scores, indent=2))

    np.save(out_dir / "oof.npy", oof)
    (out_dir / "scores.json").write_text(json.dumps(scores, indent=2))
    ids = [i for g in test_groups for i in g["ids"]]
    preds = [p for arr in test_preds for p in arr]
    to_submission(ids, preds).to_csv(out_dir / "submission.csv", index=False)
    print(f"-> {out_dir}")


if __name__ == "__main__":
    main()
