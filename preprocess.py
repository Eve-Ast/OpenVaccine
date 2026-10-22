"""Préparer les données : python preprocess.py.

Lit les données Kaggle, encode les entrées et conserve les folds du dépôt.
Écrit train.npz, test_107.npz et test_130.npz dans data/processed/.
Le filtre SN_filter est appliqué ensuite dans train.py, sans modifier les cibles.
"""
import numpy as np

from src import config as C
from src.preprocessing.data import build_errors, build_inputs, build_targets, load_test, load_train
from src.preprocessing.folds import load_folds


def check_sequences(df):
    """Vérifier les identifiants, longueurs et caractères avant l'encodage."""
    if df.id.duplicated().any():
        raise ValueError('Identifiants de molécules dupliqués')
    for column, vocabulary in [('sequence', C.SEQ_VOCAB), ('structure', C.STRUCT_VOCAB),
                               ('predicted_loop_type', C.LOOP_VOCAB)]:
        for text, length in zip(df[column], df.seq_length):
            if not isinstance(text, str) or len(text) != length or not set(text) <= set(vocabulary):
                raise ValueError(f'{column} : longueur ou caractères incorrects')


def main():
    train, test = load_train(), load_test()
    check_sequences(train)
    check_sequences(test)
    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Les cinq cibles restent dans l'ordre défini par config.TARGETS.
    targets, errors = build_targets(train), build_errors(train)
    if targets.shape != (len(train), C.SEQ_SCORED, len(C.TARGETS)) or not np.isfinite(targets).all():
        raise ValueError('Cibles manquantes ou dimensions incorrectes')
    folds = {}
    for name in ['kfold', 'group']:
        assignments = load_folds(name).set_index('id')['fold']
        folds[f'fold_{name}'] = assignments.loc[train.id].to_numpy('int8')

    np.savez_compressed(
        C.PROCESSED_DIR / 'train.npz', ids=train.id.to_numpy().astype(str),
        **build_inputs(train), targets=targets, errors=errors, **folds,
        signal_to_noise=train.signal_to_noise.to_numpy('float32'),
        sn_filter=train.SN_filter.to_numpy('int8'))
    print(f'train.npz : {len(train)} ARN, dont {(train.SN_filter == 1).sum()} fiables')

    # Une longueur par fichier : aucun padding nécessaire.
    for length, group in test.groupby('seq_length'):
        np.savez_compressed(C.PROCESSED_DIR / f'test_{length}.npz',
                            ids=group.id.to_numpy().astype(str), **build_inputs(group))
        print(f'test_{length}.npz : {len(group)} ARN')


if __name__ == '__main__':
    main()
