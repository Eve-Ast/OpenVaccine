"""Stratégie 2 : Transfer Learning (pré-entraînement MLM -> fine-tuning), en PyTorch.

src.evaluation.metrics importe Keras : si son backend est TensorFlow, on l'empêche
de réserver toute la mémoire GPU, qui doit rester disponible pour PyTorch.
"""
import os

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
