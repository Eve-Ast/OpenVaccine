"""Conversion des prédictions au format Kaggle (id_seqpos + 5 colonnes)."""
import pandas as pd

from .. import config as C


def to_submission(ids, preds):
    """ids : liste d'id ; preds : liste de tableaux (L, 5), un par molécule."""
    rows = []
    for mol_id, p in zip(ids, preds):
        for pos in range(len(p)):
            rows.append([f"{mol_id}_{pos}", *p[pos]])
    sub = pd.DataFrame(rows, columns=["id_seqpos", *C.TARGETS])
    # même ordre de lignes que le sample_submission
    order = pd.read_csv(C.DATA_DIR / "sample_submission.csv", usecols=["id_seqpos"])
    return order.merge(sub, on="id_seqpos", how="left")
