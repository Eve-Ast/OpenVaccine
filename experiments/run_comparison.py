"""Expériences contrôlées sans régénération des folds ni écrasement des résultats.

python experiments/run_comparison.py --five-fold --reuse-pilots --jobs 2
Les pilotes utilisent le fold 0. Le test n'intervient pas dans la sélection.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MATRIX = [
    ("gru", "seq", {"hidden": 64, "n_layers": 2, "dropout": .2}),
    ("gru", "annot", {"hidden": 64, "n_layers": 2, "dropout": .2}),
    ("transformer", "annot", {"d_model": 64, "num_heads": 4, "n_layers": 2, "ff_dim": 128, "dropout": .2}),
    ("transformer", "graph", {"d_model": 64, "num_heads": 4, "n_layers": 2, "ff_dim": 128, "dropout": .2}),
    ("gcn", "annot", {"units": 64, "n_layers": 4, "dropout": .2}),
    ("gcn", "graph", {"units": 64, "n_layers": 4, "dropout": .2}),
    ("conv_gru", "annot", {"hidden": 64, "n_layers": 2, "dropout": .2, "filters": 64, "kernel_size": 5}),
    ("attention_gru", "annot", {"hidden": 64, "n_layers": 2, "dropout": .2, "num_heads": 4}),
    ("gat", "annot", {"units": 64, "num_heads": 4, "n_layers": 4, "dropout": .2}),
    ("gat", "graph", {"units": 64, "num_heads": 4, "n_layers": 4, "dropout": .2}),
]


def run_config(entry, args, env):
    model, level, params = entry
    name = f"{'cv' if args.five_fold else 'pilot'}_{model}_{level}"
    out = ROOT / 'outputs' / name
    if (out/'scores.json').exists():
        saved = json.loads((out/'scores.json').read_text())
        expected = set(map(str, range(5))) if args.five_fold else {'0'}
        if (set(saved['fold_mcrmse']) == expected and
            saved['args']['epochs'] == args.epochs and
            json.loads(saved['args']['params']) == params):
            print(f'Already completed: {name}', flush=True)
            return name
        raise RuntimeError(f'Run incomplet ou configuration différente : {out}')
    command = [sys.executable, '-B', 'train.py', '--model', model, '--level', level,
               '--params', json.dumps(params), '--name', name, '--epochs', str(args.epochs),
               '--patience', '8', '--batch-size', '32', '--optimizer', 'adamw', '--lr', '0.001', '--skip-test']
    if not args.five_fold:
        command += ['--only-fold', '0']
    elif args.reuse_pilots:
        command += ['--reuse-pilot', f'pilot_{model}_{level}']
    print(f'Starting {name}', flush=True)
    with (ROOT/'outputs'/f'{name}.log').open('w') as log:
        subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    print(f'Completed {name}', flush=True)
    return name


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--five-fold', action='store_true')
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--jobs', type=int, choices=[1, 2], default=1)
    p.add_argument('--reuse-pilots', action='store_true', help='réutiliser les folds 0 compatibles')
    p.add_argument('--select', nargs='+', choices=[f'{m}_{l}' for m,l,_ in MATRIX])
    args = p.parse_args()
    env = dict(os.environ, MPLCONFIGDIR='/tmp/openvaccine-mpl')
    selected = [x for x in MATRIX if not args.select or f'{x[0]}_{x[1]}' in args.select]
    # Les tâches longues commencent en premier ; chaque entraînement reste isolé.
    cost = {'attention_gru': 4, 'conv_gru': 3, 'gru': 2, 'gat': 1, 'transformer': 0, 'gcn': 0}
    if args.jobs > 1:
        selected.sort(key=lambda x: cost[x[0]], reverse=True)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(run_config, x, args, env) for x in selected]
        for f in as_completed(futures):
            f.result()
            subprocess.run([sys.executable, '-B', 'experiments/summarize.py'], cwd=ROOT, env=env, check=True)


if __name__ == '__main__':
    main()
