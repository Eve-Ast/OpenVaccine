"""Pré-entraînement MLM de l'encodeur (Stratégie 2, Personne C).

    python -m experiments.transfer.pretrain                         # 60 epochs, masquage 15 %
    python -m experiments.transfer.pretrain --span 3 --epochs 80    # span masking
    python -m experiments.transfer.pretrain --epochs 2              # test rapide

Sorties dans outputs/mlm_pretrain/ : encoder.pt (meilleure val_loss), history.csv
"""
import argparse

from src.transfer.mlm import DEFAULT_CKPT, pretrain


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--mask-p", type=float, default=0.15, help="proportion de bases masquées")
    p.add_argument("--joint-p", type=float, default=0.5, help="proba de masquer aussi structure + boucle")
    p.add_argument("--span", type=int, default=1, help="longueur (impaire) des spans masqués")
    p.add_argument("--out", default=str(DEFAULT_CKPT))
    return p.parse_args()


if __name__ == "__main__":
    pretrain(parse_args())
