"""Produire uniquement les trois figures du rapport à partir des CV terminées."""
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('MPLCONFIGDIR', '/tmp/openvaccine-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.run_comparison import MATRIX
from src import config as C

LABELS = {'gru': 'BiGRU', 'conv_gru': 'Conv1D + BiGRU', 'attention_gru': 'BiGRU + Attention',
          'transformer': 'Transformer', 'gcn': 'GCN', 'gat': 'GAT'}


def load_results():
    rows = []
    for model, level, _ in MATRIX:
        path = C.OUTPUT_DIR / f'cv_{model}_{level}' / 'scores.json'
        if not path.exists():
            continue
        score = json.loads(path.read_text())
        if set(score['fold_mcrmse']) != set(map(str, range(C.N_FOLDS))):
            continue
        rows.append(dict(model=model, level=level, mean=score['fold_mean'],
                         std=score['fold_std'], oof=score['cv_mcrmse'],
                         **{t: score['rmse_per_target'][t] for t in C.SCORED_TARGETS}))
    return pd.DataFrame(rows)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(ROOT / 'figures' / name, dpi=170)
    plt.close(fig)


def plot_architectures(results):
    data = results[results.level == 'annot'].sort_values('mean')
    if data.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.barh([LABELS[m] for m in data.model], data['mean'], xerr=data['std'],
            color='#5886a5', capsize=4)
    ax.invert_yaxis()
    for i, row in enumerate(data.itertuples()):
        ax.text(row.mean + row.std + .005, i, f'{row.mean:.4f} ± {row.std:.4f}', va='center', fontsize=9)
    ax.set_xlim(0, float((data['mean'] + data['std']).max()) + .10)
    ax.set(xlabel='MCRMSE : moyenne ± écart-type des cinq folds',
           title='Comparaison des architectures — mêmes entrées annot')
    save(fig, 'cv_architectures.png')


def plot_structure(results):
    data = results.pivot(index='model', columns='level', values='oof')
    data = data.reindex(index=LABELS, columns=['seq', 'annot', 'graph']).dropna(how='all')
    data.index = [LABELS[m] for m in data.index]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    data.plot.bar(ax=ax, color=['#aab4bd', '#5886a5', '#db9050'], rot=20)
    ax.set(ylabel='MCRMSE sur les prédictions OOF', xlabel='',
           title="Contribution des informations structurales")
    ax.legend(title='Entrées')
    save(fig, 'cv_structure.png')


def plot_targets(results):
    data = results.copy()
    data.index = [f'{LABELS[m]} ({level})' for m, level in zip(data.model, data.level)]
    fig, ax = plt.subplots(figsize=(12, 5))
    data[C.SCORED_TARGETS].plot.bar(ax=ax, rot=35)
    ax.set(ylabel='RMSE sur les prédictions OOF', xlabel='', title='Erreur par cible expérimentale')
    ax.legend(fontsize=9, loc='upper center', ncol=3)
    save(fig, 'cv_targets.png')


def main():
    results = load_results()
    if results.empty:
        print('Aucune validation croisée à cinq folds terminée.')
        return
    (ROOT / 'figures').mkdir(exist_ok=True)
    plot_architectures(results)
    plot_structure(results)
    plot_targets(results)
    print(f'Trois figures mises à jour dans figures/ ({len(results)} configurations).')


if __name__ == '__main__':
    main()
