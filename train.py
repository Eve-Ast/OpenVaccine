"""Entraînement en validation croisée, commun à tous les modèles.

Exemples :
    python train.py --model gru
    python train.py --model conv_gru --optimizer rmsprop --lr 3e-4 --batch-size 32
    python train.py --model transformer --params '{"use_bpps": false}' --name transformer_no_bpps
    python train.py --model gnn --only-fold 0 --epochs 5        # test rapide

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
from src.data import build_inputs, build_targets, load_test, load_train, select
from src.folds import load_folds
from src.metrics import mcrmse, mcrmse_loss
from src.models import MODELS
from src.submission import to_submission

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
    p.add_argument("--model", required=True, choices=MODELS)
    p.add_argument("--name", help="nom du dossier de sortie (défaut : nom du modèle)")
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
    name = args.name or args.model
    out_dir = C.OUTPUT_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    keras.utils.set_random_seed(C.SEED)

    train = load_train().merge(load_folds(args.folds), on="id")
    test = load_test()
    X, Y = build_inputs(train), build_targets(train)
    n_feats = X["feats"].shape[-1]
    test_groups = [(g, build_inputs(g)) for _, g in test.groupby("seq_length")]
    clean = (train.SN_filter == 1).to_numpy()

    folds = [args.only_fold] if args.only_fold is not None else range(C.N_FOLDS)
    oof = np.full_like(Y, np.nan)
    test_preds = [np.zeros((len(g), g.seq_length.iloc[0], len(C.TARGETS))) for g, _ in test_groups]
    fold_scores = {}

    for k in folds:
        print(f"\n===== {name} | fold {k} =====")
        is_val = (train.fold == k).to_numpy()
        tr = ~is_val & (True if args.all_samples else clean)
        va = is_val & clean  # validation toujours sur les molécules fiables

        model = MODELS[args.model](n_feats, **json.loads(args.params))
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

        for i, (_, Xg) in enumerate(test_groups):
            test_preds[i] += model.predict(select(Xg, slice(None), keys), batch_size=256, verbose=0) / len(folds)

    done = np.isin(train.fold, list(folds)) & clean
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
    ids = [i for g, _ in test_groups for i in g.id]
    preds = [p for arr in test_preds for p in arr]
    to_submission(ids, preds).to_csv(out_dir / "submission.csv", index=False)
    print(f"-> {out_dir}")


if __name__ == "__main__":
    main()
