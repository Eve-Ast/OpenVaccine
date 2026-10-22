"""Pré-entraînement auto-supervisé par masquage (MLM) sur train + test_107 + test_130."""
import math
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from src import config as C
from .data import PROCESSED, load_inputs, iterate_batches, n_batches
from .model import RNAEncoder, MLMModel, pairs_to_adj, OFFSETS, N_CLASSES, MASK_IDS, CHANNELS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PRETRAIN_DIR = C.OUTPUT_DIR / "mlm_pretrain"
DEFAULT_CKPT = PRETRAIN_DIR / "encoder.pt"


def seed_all(s=C.SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def apply_mlm_mask(b, p=0.15, p_joint=0.5, span=1):
    """BERT 80/10/10 sur la base. Avec proba p_joint, structure et boucle sont masquées aux
    mêmes positions ; on efface alors aussi ce qui trahirait la structure : adjacence
    (lignes/colonnes) et feats (pair_distance vaut 0 ssi la base est non appariée).
    La BPP reste visible : apprendre la complémentarité est l'objectif."""
    tok = b["tokens"].long()
    B, L, _ = tok.shape
    dev = tok.device
    sel = torch.rand(B, L, device=dev) < p / span
    if span > 1:   # span impair : masque des motifs contigus
        sel = F.max_pool1d(sel.float()[:, None], span, 1, padding=span // 2)[:, 0].bool()
    joint = sel & (torch.rand(B, L, device=dev) < p_joint)

    new, labels = tok.clone(), {}
    r = torch.rand(B, L, device=dev)
    new[..., 0][sel & (r < 0.8)] = MASK_IDS[0]
    rnd = sel & (r >= 0.8) & (r < 0.9)
    new[..., 0][rnd] = OFFSETS[0] + torch.randint(0, N_CLASSES[0], (int(rnd.sum()),), device=dev)
    labels["seq"] = torch.where(sel, tok[..., 0] - OFFSETS[0], -100)
    for c in (1, 2):
        new[..., c][joint] = MASK_IDS[c]
        labels[CHANNELS[c]] = torch.where(joint, tok[..., c] - OFFSETS[c], -100)

    keep = (~joint).float()
    nb = dict(b)
    nb["tokens"] = new
    nb["adj"] = pairs_to_adj(b["pairs"].long()) * keep[:, :, None] * keep[:, None, :]
    nb["feats"] = b["feats"] * keep[..., None]
    return nb, labels


def mlm_loss(out, labels, w=None):
    w = w or {"seq": 1.0, "struct": 0.5, "loop": 0.5}
    tot = 0.0
    for k, logits in out.items():
        if (labels[k] != -100).any():
            tot = tot + w[k] * F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                               labels[k].reshape(-1), ignore_index=-100)
    return tot


def warmup_cosine(opt, total, warmup_frac=0.06):
    wu = max(1, int(total * warmup_frac))
    f = lambda s: (s + 1) / wu if s < wu else 0.5 * (1 + math.cos(math.pi * (s - wu) / max(1, total - wu)))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


def pretrain(args):
    seed_all()
    PRETRAIN_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(C.SEED)
    tr_groups, va_groups = [], []
    for name in PROCESSED:                                   # 5 % de chaque groupe en hold-out
        g = load_inputs(name)
        perm = rng.permutation(len(g["tokens"]))
        n_val = max(1, int(0.05 * len(perm)))
        va_groups.append({k: v[perm[:n_val]] for k, v in g.items()})
        tr_groups.append({k: v[perm[n_val:]] for k, v in g.items()})
    print(f"MLM : {sum(len(g['tokens']) for g in tr_groups)} molécules train, "
          f"{sum(len(g['tokens']) for g in va_groups)} validation")

    model = MLMModel(RNAEncoder()).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = warmup_cosine(opt, args.epochs * n_batches(tr_groups, args.batch_size))
    best, history = float("inf"), []
    for ep in range(args.epochs):
        model.train(); tl, nb = 0.0, 0
        for b in iterate_batches(tr_groups, args.batch_size, shuffle=True, rng=rng, device=DEVICE):
            mb, lab = apply_mlm_mask(b, args.mask_p, args.joint_p, args.span)
            loss = mlm_loss(model(mb), lab)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
            tl += loss.item(); nb += 1

        model.eval(); torch.manual_seed(0)                   # masque de validation reproductible
        vl, nv, corr, tot = 0.0, 0, 0, 0
        with torch.no_grad():
            for b in iterate_batches(va_groups, args.batch_size, device=DEVICE):
                mb, lab = apply_mlm_mask(b, args.mask_p, args.joint_p, args.span)
                out = model(mb); vl += mlm_loss(out, lab).item(); nv += 1
                m = lab["seq"] != -100
                corr += (out["seq"].argmax(-1)[m] == lab["seq"][m]).sum().item(); tot += m.sum().item()
        torch.manual_seed(C.SEED + ep + 1)
        row = {"epoch": ep, "loss": tl / nb, "val_loss": vl / nv, "masked_nt_acc": corr / max(tot, 1)}
        history.append(row)
        print(f"[MLM] ep {ep:3d}  loss {row['loss']:.4f}  val_loss {row['val_loss']:.4f}  "
              f"masked_nt_acc {row['masked_nt_acc']:.3f}")
        if row["val_loss"] < best:
            best = row["val_loss"]
            torch.save(model.enc.state_dict(), args.out)
        pd.DataFrame(history).to_csv(PRETRAIN_DIR / "history.csv", index=False)
    print(f"-> {args.out}")
