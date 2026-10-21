"""Diagnostic sur OOF fiables uniquement, jamais sur les données d'entraînement.

python experiments/prediction_plots.py --runs cv_gru_annot pilot_conv_gru_annot
Les exemples de profils sont les molécules aux percentiles 50 et 90 de
l'erreur par molécule, déterminées sans sélection visuelle.
"""
import argparse
import json
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/openvaccine-mpl')
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--runs', nargs='+', required=True)
    args=p.parse_args()
    with np.load(ROOT/'data/processed/train.npz', allow_pickle=False) as z:
        targets=z['targets']; clean=z['sn_filter']==1; ids=z['ids']
        folds={key.removeprefix('fold_'):z[key] for key in z.files if key.startswith('fold_')}
    # Lire les constantes ne charge pas le backend Keras.
    import sys
    sys.path.insert(0,str(ROOT))
    from src import config as C
    out=ROOT/'outputs/comparison'
    out.mkdir(exist_ok=True)
    for run in args.runs:
        path=ROOT/'outputs'/run
        scores=json.loads((path/'scores.json').read_text())
        fold=folds[scores['args']['folds']]
        pred=np.load(path/'oof.npy')
        coverage=np.isin(fold, [int(k) for k in scores['fold_mcrmse']])
        mask=clean&coverage
        if not np.isfinite(pred[mask]).all():
            raise ValueError(f'OOF incomplets : {run}')
        y=targets[mask][...,C.SCORED_IDX]; yhat=pred[mask][...,C.SCORED_IDX]
        metrics=[]
        fig,axes=plt.subplots(1,3,figsize=(13,4))
        for j,t in enumerate(C.SCORED_TARGETS):
            a,b=y[...,j].ravel(),yhat[...,j].ravel()
            mse=float(np.mean((a-b)**2)); mae=float(np.mean(np.abs(a-b)))
            r2=1-float(np.sum((a-b)**2))/float(np.sum((a-a.mean())**2))
            metrics.append(dict(target=t,rmse=np.sqrt(mse),mae=mae,r2=r2,n_sequences=int(mask.sum()),n_positions=len(a)))
            ax=axes[j]
            hb=ax.hexbin(a,b,gridsize=55,bins='log',mincnt=1,cmap='viridis')
            lo=min(float(a.min()),float(b.min())); hi=max(float(a.max()),float(b.max()))
            ax.plot([lo,hi],[lo,hi],'r--',linewidth=1,label='Prédiction parfaite')
            ax.set(xlabel='Valeur réelle',ylabel='Valeur prédite',title=f'{t}\nRMSE={np.sqrt(mse):.3f}, MAE={mae:.3f}, R²={r2:.3f}')
            ax.legend(fontsize=7); fig.colorbar(hb,ax=ax,label='Nombre de positions (log)')
        fig.suptitle(f'{run} — prédictions OOF, {mask.sum()} molécules fiables')
        fig.tight_layout(); fig.savefig(out/f'{run}_actual_predicted.png',dpi=170); plt.close(fig)
        pd.DataFrame(metrics).to_csv(out/f'{run}_prediction_metrics.csv',index=False)
        errors=np.mean((y-yhat)**2,axis=(1,2))
        order=np.argsort(errors,kind='stable')
        selected=[order[len(order)//2],order[int(.9*(len(order)-1))]]
        fig,axes=plt.subplots(2,3,figsize=(13,6),sharex=True)
        examples=[]
        for r,(i,label) in enumerate(zip(selected,['Erreur médiane','Erreur au percentile 90'])):
            mol=str(ids[mask][i]); examples.append(dict(id=mol,selection=label,fold=int(fold[mask][i])))
            for j,t in enumerate(C.SCORED_TARGETS):
                ax=axes[r,j]; ax.plot(y[i,:,j],label='Réel',linewidth=1.3); ax.plot(yhat[i,:,j],label='Prédit',linewidth=1.3)
                ax.set(title=f'{label} — {mol}\n{t}',xlabel='Position du nucléotide',ylabel='Valeur'); ax.legend(fontsize=7)
        fig.suptitle(f'{run} — exemples OOF sélectionnés par erreur, sans choix manuel')
        fig.tight_layout(); fig.savefig(out/f'{run}_profiles.png',dpi=170); plt.close(fig)
        (out/f'{run}_profile_examples.json').write_text(json.dumps(examples,indent=2))
        print(run, metrics,flush=True)


if __name__=='__main__':
    main()
