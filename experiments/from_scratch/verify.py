"""Vérifier les scores à partir des prédictions OOF et des folds enregistrés."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src import config as C
from experiments.from_scratch.run import MATRIX


def score(targets, predictions):
    # Calcul indépendant de la fonction utilisée pendant l'entraînement.
    error = targets[..., C.SCORED_IDX] - predictions[..., C.SCORED_IDX]
    return float(np.sqrt(np.mean(error ** 2, axis=(0, 1))).mean())


def main():
    with np.load(C.PROCESSED_DIR / 'train.npz', allow_pickle=False) as data:
        targets = data['targets']
        clean = data['sn_filter'] == 1
        ids = data['ids']
        assignments = {name: data[f'fold_{name}'] for name in ['kfold', 'group']}
    for name, folds in assignments.items():
        csv = pd.read_csv(C.FOLDS_DIR / f'folds_{name}.csv').set_index('id')
        np.testing.assert_array_equal(folds, csv.loc[ids, 'fold'].to_numpy())

    report = []
    reference_protocol = None
    protocol_keys = ['folds', 'optimizer', 'lr', 'batch_size', 'epochs', 'patience',
                     'seed', 'all_samples', 'skip_test']
    for model, level, params in MATRIX:
        name = f'cv_{model}_{level}'
        path = C.OUTPUT_DIR / name
        saved = json.loads((path / 'scores.json').read_text())
        args = saved['args']
        assert args['model'] == model and args['level'] == level, name
        assert json.loads(args['params']) == params, name
        protocol = {key: args[key] for key in protocol_keys}
        if reference_protocol is None:
            reference_protocol = protocol
        assert protocol == reference_protocol, f'Protocole différent : {name}'
        assert set(saved['fold_mcrmse']) == set(map(str, range(C.N_FOLDS))), name

        folds = assignments[args['folds']]
        predictions = np.load(path / 'oof.npy')
        assert predictions.shape == targets.shape and np.isfinite(predictions).all(), name
        np.testing.assert_allclose(score(targets[clean], predictions[clean]), saved['cv_mcrmse'], atol=1e-7, rtol=0)
        fold_scores = []
        for fold in range(C.N_FOLDS):
            mask = clean & (folds == fold)
            value = score(targets[mask], predictions[mask])
            fold_scores.append(value)
            np.testing.assert_allclose(value, saved['fold_mcrmse'][str(fold)], atol=1e-7, rtol=0)
            history = pd.read_csv(path / f'history_fold{fold}.csv')
            np.testing.assert_allclose(value, history.val_mcrmse.min(), atol=1e-7, rtol=0)
            assert (path / f'fold{fold}.weights.h5').is_file(), name
        np.testing.assert_allclose([np.mean(fold_scores), np.std(fold_scores, ddof=1)],
                                   [saved['fold_mean'], saved['fold_std']], atol=1e-7, rtol=0)

        # Certains anciens runs ont réutilisé leur pilote : contrôler sa provenance.
        source = saved.get('reused_pilot')
        if source:
            original = C.OUTPUT_DIR / source
            np.testing.assert_array_equal(predictions[folds == 0], np.load(original / 'oof.npy')[folds == 0])
            for filename in ['fold0.weights.h5', 'history_fold0.csv']:
                assert hashlib.sha256((path / filename).read_bytes()).digest() == hashlib.sha256((original / filename).read_bytes()).digest()
        report.append(dict(run=name, folds=5, mcrmse=saved['cv_mcrmse'], reused_pilot=source))
        print(name, 'verified', flush=True)

    result = dict(runs=report, folds_verified=len(report) * C.N_FOLDS,
                  fold_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in C.FOLDS_DIR.glob('*.csv')})
    destination = C.OUTPUT_DIR / 'comparison'
    destination.mkdir(exist_ok=True)
    (destination / 'verification.json').write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
