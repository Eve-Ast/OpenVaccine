"""MCRMSE : métrique Kaggle (numpy) et loss d'entraînement (Keras)."""
import numpy as np
from keras import ops

from . import config as C


def mcrmse(y_true, y_pred, cols=C.SCORED_IDX):
    """Métrique officielle : moyenne des RMSE des colonnes évaluées.

    y_true, y_pred : (N, 68, 5)
    """
    y_true, y_pred = y_true[..., cols], y_pred[..., cols]
    rmse_per_col = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=(0, 1)))
    return float(np.mean(rmse_per_col))


def mcrmse_loss(y_true, y_pred):
    """Loss masquée : le modèle prédit L positions, on ne garde que les 68 annotées.

    On entraîne sur les 5 cibles (multi-tâche) ; la métrique n'en garde que 3.
    """
    y_pred = y_pred[:, : C.SEQ_SCORED]
    mse_per_col = ops.mean(ops.square(y_true - y_pred), axis=(0, 1))
    return ops.mean(ops.sqrt(mse_per_col + 1e-8))
