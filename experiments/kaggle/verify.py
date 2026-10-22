"""Vérifier les cinq folds Kaggle et comparer leur score à la baseline BiGRU."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/openvaccine-mpl')
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src import config as C
from src.preprocessing.data import load_processed
from src.evaluation.metrics import mcrmse


def main():
    run = C.OUTPUT_DIR/'cv_kaggle_ae_common'
    summary = json.loads((run/'scores.json').read_text())
    if not summary['complete'] or set(summary['fold_mcrmse']) != set(map(str, range(5))):
        raise ValueError('Five completed non-smoke folds required')
    data = load_processed('train')
    clean, fold = data['sn_filter']==1, data['fold_kfold']
    oof = np.load(run/'oof.npy')
    for k in range(5):
        record = json.loads((run/f'fold{k}.json').read_text())
        va, tr = clean & (fold==k), clean & (fold!=k)
        assert record['validation_ids'] == data['ids'][va].tolist()
        assert record['train_ids'] == data['ids'][tr].tolist()
        np.testing.assert_allclose(mcrmse(data['targets'][va], oof[va]), summary['fold_mcrmse'][str(k)], atol=1e-7)
    (run/'verification.json').write_text(json.dumps({'verified_folds':5, 'cv_mcrmse':mcrmse(data['targets'][clean],oof[clean])},indent=2))
    np.testing.assert_allclose(mcrmse(data['targets'][clean], oof[clean]), summary['cv_mcrmse'], atol=1e-7, rtol=0)
    values = list(summary['fold_mcrmse'].values())
    np.testing.assert_allclose([np.mean(values), np.std(values, ddof=1)],
                               [summary['fold_mean'], summary['fold_std']], atol=1e-7, rtol=0)
    baseline = json.loads((C.OUTPUT_DIR / 'cv_gru_annot' / 'scores.json').read_text())
    fig, ax = plt.subplots(figsize=(7, 4.5))
    means = [baseline['fold_mean'], summary['fold_mean']]
    stds = [baseline['fold_std'], summary['fold_std']]
    ax.bar(['BiGRU annot', 'Kaggle : AE + GNN + attention + CNN'], means,
           yerr=stds, capsize=5, color=['#5886a5', '#db9050'])
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax.text(i, mean + std + .007, f'{mean:.4f} ± {std:.4f}', ha='center')
    ax.set(ylabel='MCRMSE : moyenne ± écart-type des cinq folds', ylim=(0, max(means) + .06),
           title='Même filtre de qualité et mêmes folds')
    fig.text(.5, .02, 'Caractéristiques et budgets différents ; AE préentraîné sur le train de chaque fold.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .05, 1, 1))
    destination = C.ROOT / 'figures' / 'kaggle'
    destination.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination / 'cv_comparison.png', dpi=170)
    plt.close(fig)
    print(f"Kaggle : cinq folds vérifiés, MCRMSE = {summary['fold_mean']:.5f} ± {summary['fold_std']:.5f}")


if __name__ == '__main__':
    main()
