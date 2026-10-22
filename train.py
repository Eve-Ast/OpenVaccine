"""Validation croisée commune : données/folds existants, sorties compatibles.

Exemple : python train.py --model gru --level annot --only-fold 0 --name gru_pilot
Les checkpoints sont des poids Keras (.weights.h5). scores.json contient
les paramètres effectifs nécessaires pour reconstruire le modèle.
"""
import argparse
import inspect
import json
import os
import time

os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "4")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")

import keras
import numpy as np
import tensorflow as tf

from src import config as C
from src.evaluation.metrics import mcrmse, mcrmse_loss
from src.evaluation.submission import to_submission
from src.models import MODELS
from src.models.common import LEVELS
from src.models.mean import MeanBaseline
from src.preprocessing.data import inputs_of, load_processed, select

OPTIMIZERS = {
    "adam": lambda lr: keras.optimizers.Adam(lr),
    "adamw": lambda lr: keras.optimizers.AdamW(lr, weight_decay=1e-4),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["mean", *MODELS])
    p.add_argument("--name", help="dossier de sortie unique")
    p.add_argument("--level", choices=LEVELS)
    p.add_argument("--params", default="{}", help="hyperparamètres en JSON")
    p.add_argument("--folds", default="kfold", choices=["kfold", "group"])
    p.add_argument("--only-fold", type=int, choices=range(C.N_FOLDS))
    p.add_argument("--optimizer", default="adamw", choices=OPTIMIZERS)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--seed", type=int, default=C.SEED)
    p.add_argument("--skip-test", action="store_true", help="sans inférence sur le test Kaggle")
    p.add_argument("--all-samples", action="store_true")
    args = p.parse_args()
    if min(args.batch_size, args.epochs, args.patience) < 1 or args.lr <= 0:
        p.error("batch-size, epochs, patience et lr doivent être positifs")
    return args


def dataset(x, y, batch_size, shuffle=False, seed=42):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(len(y), seed=seed, reshuffle_each_iteration=True)
    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    return ds.batch(batch_size).with_options(options).prefetch(1)


def predict(model, x, batch_size):
    # Pas de tf.data supplémentaire : prédictions par petits lots en mode inférence.
    n = len(next(iter(x.values())))
    return np.concatenate([
        np.asarray(model({k: v[i:i + batch_size] for k, v in x.items()}, training=False))
        for i in range(0, n, batch_size)
    ])


class ValidationScore(keras.callbacks.Callback):
    """MCRMSE global sur les trois cibles, sans moyenne de RMSE par minibatch."""
    def __init__(self, x, y, batch_size):
        super().__init__()
        self.x, self.y, self.batch_size = x, y, batch_size

    def on_epoch_end(self, epoch, logs=None):
        value = mcrmse(self.y, predict(self.model, self.x, self.batch_size)[:, :C.SEQ_SCORED])
        if not np.isfinite(value):
            raise FloatingPointError("validation MCRMSE non fini")
        logs["val_mcrmse"] = value


def main():
    args = parse_args()
    params = json.loads(args.params)
    if args.level:
        params["level"] = args.level
    effective = {} if args.model == "mean" else {
        k: v.default for k, v in inspect.signature(MODELS[args.model]).parameters.items()
        if k != "n_feats" and v.default is not inspect.Parameter.empty
    }
    effective.update(params)
    level = effective.get("level")
    name = args.name or "_".join([args.model] + ([level] if level else []))
    out_dir = C.OUTPUT_DIR / name
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"{out_dir} non vide : choisir un nouveau --name")
    out_dir.mkdir(parents=True, exist_ok=True)
    tf.config.experimental.enable_op_determinism()
    started = time.perf_counter()
    train = load_processed("train")
    X, Y = inputs_of(train), train["targets"]
    assert Y.shape == (len(train["ids"]), C.SEQ_SCORED, len(C.TARGETS))
    assert np.isfinite(Y).all()
    fold = train[f"fold_{args.folds}"]
    clean = train["sn_filter"] == 1
    n_feats = X["feats"].shape[-1]
    folds = [args.only_fold] if args.only_fold is not None else list(range(C.N_FOLDS))
    test_groups = [] if args.skip_test else [load_processed(f"test_{L}") for L in (107, 130)]
    test_preds = [np.zeros((*g["pairs"].shape, len(C.TARGETS)), dtype="float32") for g in test_groups]
    oof = np.full_like(Y, np.nan)
    fold_scores, details = {}, {}
    (out_dir / "config.json").write_text(json.dumps({"args": vars(args), "model_params": effective}, indent=2))

    for k in folds:
        keras.backend.clear_session()
        keras.utils.set_random_seed(args.seed)
        fold_started = time.perf_counter()
        print(f"\n===== {name} | fold {k} =====", flush=True)
        is_val = fold == k
        tr = ~is_val & (True if args.all_samples else clean)
        va = is_val & clean
        if not tr.any() or not va.any():
            raise ValueError("fold sans exemples d'entraînement ou de validation")
        if args.model == "mean":
            model, keys = MeanBaseline(Y[tr]), ["pairs"]
            inference = lambda x: model.predict(x)
            extra = {"trainable_parameters": 0}
        else:
            model = MODELS[args.model](n_feats, **params)
            keys = [t.name for t in model.inputs]
            model.compile(optimizer=OPTIMIZERS[args.optimizer](args.lr), loss=mcrmse_loss)
            xv = select(X, va, keys)
            history = model.fit(
                dataset(select(X, tr, keys), Y[tr], args.batch_size, True, args.seed),
                validation_data=dataset(xv, Y[va], args.batch_size),
                epochs=args.epochs, verbose=2, shuffle=False,
                callbacks=[
                    ValidationScore(xv, Y[va], args.batch_size),
                    keras.callbacks.ModelCheckpoint(out_dir / f"fold{k}.weights.h5", monitor="val_mcrmse", mode="min", save_best_only=True, save_weights_only=True),
                    keras.callbacks.EarlyStopping(monitor="val_mcrmse", mode="min", patience=args.patience),
                    keras.callbacks.ReduceLROnPlateau(monitor="val_mcrmse", mode="min", factor=0.5, patience=max(1, args.patience // 2)),
                    keras.callbacks.CSVLogger(out_dir / f"history_fold{k}.csv"),
                    keras.callbacks.TerminateOnNaN(),
                ],
            )
            # Recharge explicitement le checkpoint, y compris si epochs est atteint.
            model.load_weights(out_dir / f"fold{k}.weights.h5")
            extra = {
                "best_epoch": int(np.argmin(history.history["val_mcrmse"]) + 1),
                "epochs_completed": len(history.epoch),
                "trainable_parameters": int(sum(np.prod(w.shape) for w in model.trainable_weights)),
                "optimizer_config": model.optimizer.get_config(),
            }
            inference = lambda x: predict(model, x, args.batch_size)
        pred = inference(select(X, is_val, keys))
        if not np.isfinite(pred).all():
            raise FloatingPointError("prédictions non finies")
        oof[is_val] = pred[:, :C.SEQ_SCORED]
        fold_scores[k] = mcrmse(Y[va], oof[va])
        rmse = np.sqrt(np.mean((Y[va] - oof[va]) ** 2, axis=(0, 1)))
        details[k] = dict(extra, n_train=int(tr.sum()), n_validation=int(va.sum()),
                          seconds=time.perf_counter() - fold_started,
                          rmse_per_target=dict(zip(C.TARGETS, rmse.tolist())))
        for i, g in enumerate(test_groups):
            test_preds[i] += inference(select(g, slice(None), keys)) / len(folds)
        done = np.isin(fold, list(fold_scores)) & clean
        per_target = np.sqrt(np.mean((Y[done] - oof[done]) ** 2, axis=(0, 1)))
        values = list(fold_scores.values())
        scores = {
            "cv_mcrmse": mcrmse(Y[done], oof[done]), "fold_mcrmse": fold_scores,
            "fold_mean": float(np.mean(values)),
            "fold_std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
            "rmse_per_target": dict(zip(C.TARGETS, per_target.tolist())),
            "fold_details": details, "model_params": effective, "n_feats": n_feats,
            "seed": args.seed, "args": vars(args),
            "duration_seconds": time.perf_counter() - started,
            "reused_pilot": None,
            "versions": {"keras": keras.__version__, "tensorflow": tf.__version__},
            "devices": [d.name for d in tf.config.list_physical_devices()],
        }
        np.save(out_dir / "oof.npy", oof)
        pending = out_dir / "scores.pending.json"
        pending.write_text(json.dumps(scores, indent=2))
        pending.replace(out_dir / "scores.json")
        print(f"fold {k} MCRMSE = {fold_scores[k]:.6f}", flush=True)
    if test_groups:
        ids = [i for g in test_groups for i in g["ids"]]
        preds = [p for arr in test_preds for p in arr]
        sub = to_submission(ids, preds)
        if not np.isfinite(sub[C.TARGETS].to_numpy()).all():
            raise FloatingPointError("soumission incomplète ou non finie")
        sub.to_csv(out_dir / "submission.csv", index=False)
    print(json.dumps(scores, indent=2), flush=True)
    print(f"-> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
