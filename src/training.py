"""Batches, prédictions et métrique de validation communs aux modèles Keras."""
import keras
import numpy as np
import tensorflow as tf
from src import config as C
from src.evaluation.metrics import mcrmse


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
