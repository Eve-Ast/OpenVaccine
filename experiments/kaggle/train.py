"""Evaluate the Kaggle AE/GNN/attention/CNN on the repository's fixed folds."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/openvaccine-mpl')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '4')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '2')
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from src import config as C
from src.evaluation.metrics import mcrmse
from src.kaggle.model import get_base, get_ae_model, get_model
from src.preprocessing.data import load_processed
from src.training import dataset, predict, ValidationScore


def features(data):
    tokens = data['tokens']
    # Ordre du notebook source : A, G, C, U ; vocabulaire commun à tous les folds.
    nucleotide = np.array([0, 2, 1, 3])[tokens[..., 0]]
    loop = tokens[..., 2] - 7
    node = np.concatenate([np.eye(4)[nucleotide], np.eye(7)[loop],
                           np.eye(28)[loop * 4 + nucleotide]], axis=-1).astype('float32')
    n, length = nucleotide.shape
    adj = np.empty((n, length, length, 5), dtype='float32')
    adj[..., 0] = data['bpps']
    adj[..., 1] = 0
    rows, positions = np.where(data['pairs'] >= 0)
    adj[rows, positions, data['pairs'][rows, positions], 1] = 1
    distance = 1 / (np.abs(np.arange(length)[:, None] - np.arange(length)) + 1)
    for channel, power in enumerate([1, 2, 4], 2):
        adj[..., channel] = distance ** power
    return {'node': node, 'adj': adj}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name', default='cv_kaggle_ae_common')
    p.add_argument('--only-fold', type=int, choices=range(5))
    p.add_argument('--ae-epochs', type=int, default=20)
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--smoke', action='store_true', help='tiny functional check, never a CV result')
    args = p.parse_args()
    if args.epochs < 1 or args.ae_epochs < 0 or args.batch_size < 1:
        p.error('invalid training budget')
    out = C.OUTPUT_DIR / args.name
    out.mkdir(parents=True, exist_ok=True)
    config_path = out / 'config.json'
    config = dict(vars(args), seed=C.SEED, protocol='fixed-kfold-sn_filter1-train-only-ae',
                  fold_sha256=hashlib.sha256((C.FOLDS_DIR/'folds_kfold.csv').read_bytes()).hexdigest(),
                  scored_targets=C.SCORED_TARGETS, keras=keras.__version__, tensorflow=tf.__version__)
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('Existing run has different configuration; choose another --name')
    config_path.write_text(json.dumps(config, indent=2))
    data = load_processed('train')
    expected = pd.read_csv(C.FOLDS_DIR/'folds_kfold.csv').set_index('id').loc[data['ids'], 'fold'].to_numpy()
    np.testing.assert_array_equal(data['fold_kfold'], expected)
    x, y = features(data), data['targets']
    clean = data['sn_filter'] == 1
    scores = {}
    oof = np.full_like(y, np.nan)
    tf.config.experimental.enable_op_determinism()
    for fold in ([args.only_fold] if args.only_fold is not None else range(5)):
        tr = np.flatnonzero(clean & (expected != fold))
        va = np.flatnonzero(clean & (expected == fold))
        if args.smoke:
            tr, va = tr[:4], va[:4]
        if (out / f'fold{fold}.json').exists():
            record = json.loads((out / f'fold{fold}.json').read_text())
            pred = np.load(out / f'pred_fold{fold}.npy')
            if pred.shape != y[va].shape or not np.isfinite(pred).all():
                raise ValueError('Invalid saved prediction')
            np.testing.assert_allclose(mcrmse(y[va], pred), record['mcrmse'], atol=1e-7)
        else:
            started = time.perf_counter()
            keras.backend.clear_session()
            keras.utils.set_random_seed(C.SEED)
            xt, xv = ({k: v[idx] for k, v in x.items()} for idx in (tr, va))
            print(f'Fold {fold}: train={len(tr)}, validation={len(va)}', flush=True)
            base = get_base()
            if args.ae_epochs:
                ae = get_ae_model(base)
                ae.fit(dataset(xt, np.zeros(len(tr), dtype='float32'), args.batch_size, True),
                       epochs=args.ae_epochs, verbose=2, shuffle=False,
                       callbacks=[keras.callbacks.CSVLogger(out / f'ae_history_fold{fold}.csv'), keras.callbacks.TerminateOnNaN()])
                base.save_weights(out / f'ae_fold{fold}.weights.h5')
                del ae
            model = get_model(base)
            model.fit(dataset(xt, y[tr], args.batch_size, True), epochs=args.epochs, verbose=2, shuffle=False,
                      callbacks=[ValidationScore(xv, y[va], args.batch_size),
                                 keras.callbacks.ModelCheckpoint(out/f'fold{fold}.weights.h5', monitor='val_mcrmse', mode='min', save_best_only=True, save_weights_only=True),
                                 keras.callbacks.EarlyStopping(monitor='val_mcrmse', mode='min', patience=8),
                                 keras.callbacks.ReduceLROnPlateau(monitor='val_mcrmse', mode='min', patience=4, factor=.5),
                                 keras.callbacks.CSVLogger(out/f'history_fold{fold}.csv'), keras.callbacks.TerminateOnNaN()])
            model.load_weights(out/f'fold{fold}.weights.h5')
            pred = predict(model, xv, args.batch_size)[:, :C.SEQ_SCORED]
            if not np.isfinite(pred).all():
                raise FloatingPointError('Non-finite prediction')
            record = dict(mcrmse=mcrmse(y[va], pred), n_train=len(tr), n_validation=len(va),
                          train_ids=data['ids'][tr].tolist(), validation_ids=data['ids'][va].tolist(),
                          seconds=time.perf_counter()-started, parameters=model.count_params())
            np.save(out/f'pred_fold{fold}.npy', pred)
            (out/f'fold{fold}.json').write_text(json.dumps(record, indent=2))
        oof[va] = pred
        scores[str(fold)] = record['mcrmse']
        done = np.isfinite(oof).all(axis=(1, 2))
        summary = dict(config=config, fold_mcrmse=scores, fold_mean=float(np.mean(list(scores.values()))),
                       fold_std=float(np.std(list(scores.values()), ddof=1)) if len(scores)>1 else None,
                       cv_mcrmse=mcrmse(y[done], oof[done]), complete=len(scores)==5 and not args.smoke)
        np.save(out/'oof.npy', oof)
        (out/'scores.json').write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary), flush=True)

    if summary['complete'] and args.name == 'cv_kaggle_ae_common':
        from experiments.kaggle.verify import main as plot_results
        plot_results()


if __name__ == '__main__':
    main()
