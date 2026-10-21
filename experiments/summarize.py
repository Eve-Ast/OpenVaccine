"""Tables et figures des expériences terminées uniquement."""
import json
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/openvaccine-mpl')
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'outputs'/'comparison'
out.mkdir(exist_ok=True)
rows=[]
for path in sorted((ROOT/'outputs').glob('*/scores.json')):
    if not path.parent.name.startswith(('pilot_','cv_')):
        continue
    s=json.loads(path.read_text())
    expected=5 if s['args']['only_fold'] is None else 1
    if len(s['fold_mcrmse']) != expected:
        continue
    row=dict(run=path.parent.name, model=s['args']['model'], level=s['model_params']['level'],
             folds=len(s['fold_mcrmse']), mcrmse=s['cv_mcrmse'], fold_mean=s['fold_mean'],
             fold_std=s['fold_std'], seconds=s['duration_seconds'],
             parameters=next(iter(s['fold_details'].values()))['trainable_parameters'])
    row.update({t:s['rmse_per_target'][t] for t in ['reactivity','deg_Mg_pH10','deg_Mg_50C']})
    rows.append(row)
    fig, axes=plt.subplots(1,2,figsize=(10,3.5))
    for h in sorted(path.parent.glob('history_fold*.csv')):
        d=pd.read_csv(h); epoch=d.epoch+1
        axes[0].plot(epoch,d.loss,label=f'{h.stem} train')
        axes[0].plot(epoch,d.val_loss,linestyle='--',label=f'{h.stem} validation')
        axes[1].plot(epoch,d.val_mcrmse,label=h.stem)
    axes[0].set(ylabel='Loss multitâche (5 cibles)',xlabel='Époque')
    axes[1].set(ylabel='MCRMSE validation (3 cibles)',xlabel='Époque')
    for a in axes: a.legend(fontsize=6); a.grid(alpha=.2)
    fig.suptitle(path.parent.name); fig.tight_layout(); fig.savefig(out/f'{path.parent.name}_learning.png',dpi=160); plt.close(fig)
if rows:
    df=pd.DataFrame(rows)
    df['error_reduction_vs_bigru_pct']=float('nan')
    for prefix in ['pilot_','cv_']:
        baseline=df[df.run.eq(prefix+'gru_annot')]
        if not baseline.empty:
            ref=float(baseline.iloc[0].mcrmse)
            same=df.run.str.startswith(prefix)&df.level.eq('annot')
            df.loc[same,'error_reduction_vs_bigru_pct']=100*(ref-df.loc[same,'mcrmse'])/ref
    df.to_csv(out/'results.csv',index=False)
    primary=df[df.run.str.startswith('cv_') & df.level.eq('annot')].sort_values('fold_mean')
    if not primary.empty:
        labels={'gru':'BiGRU', 'conv_gru':'Conv1D + BiGRU', 'attention_gru':'BiGRU + Attention',
                'transformer':'Transformer', 'gcn':'GCN', 'gat':'GAT'}
        fig,ax=plt.subplots(figsize=(9,4.5))
        positions=range(len(primary))
        ax.barh(list(positions),primary.fold_mean,xerr=primary.fold_std,
                color=['#398789' if m=='gru' else '#9db3bc' for m in primary.model],capsize=3)
        ax.set_yticks(list(positions),[labels[m] for m in primary.model])
        ax.invert_yaxis()
        ax.set(xlabel='MCRMSE — plus faible = meilleur',title='Architectures, mêmes entrées annot')
        ax.set_xlim(0,float((primary.fold_mean+primary.fold_std).max())+.08)
        for i,(_,r) in enumerate(primary.iterrows()):
            ax.text(r.fold_mean+r.fold_std+.005,i,f'{r.fold_mean:.4f} ± {r.fold_std:.4f}',va='center',fontsize=9)
        fig.text(.5,.01,'Moyenne ± écart-type des 5 folds (pas un intervalle de confiance)',ha='center',fontsize=8)
        fig.tight_layout(rect=[0,.04,1,1]);fig.savefig(out/'cv_architectures.png',dpi=170);plt.close(fig)
    lines=['# Comparaison des modèles entraînés de zéro','',
           'Protocole : KFold fixe, seed 42, SN_filter=1 en entraînement et validation, AdamW (weight_decay=1e-4), batch 32, lr initial 1e-3. Sélection du checkpoint par MCRMSE de validation. Loss sur 5 cibles et 68 positions ; score sur les 3 cibles officielles. Aucun réglage sur le test.', '',
           'BiGRU annot est la baseline architecturale. Le prédicteur par moyenne est seulement un contrôle naïf. La réduction relative de l’erreur par rapport à BiGRU est calculée uniquement pour les runs annot partageant la même couverture de folds ; ce pourcentage n’est pas une accuracy.', '',
           'Les pilotes utilisent uniquement le fold 0. Ils servent au développement et à la sélection ; leurs scores ne constituent pas une évaluation indépendante après sélection. Les résultats à cinq folds restent descriptifs sur ce même jeu de données.', '',
           '| Run | Folds | MCRMSE OOF | Moyenne folds | Écart-type folds |',
           '|---|---:|---:|---:|---:|']
    for r in rows:
        sd='—' if r['fold_std'] is None else f"{r['fold_std']:.5f}"
        lines.append(f"| {r['run']} | {r['folds']} | {r['mcrmse']:.5f} | {r['fold_mean']:.5f} | {sd} |")
    lines += ['', '## Comparaison à la baseline BiGRU (entrées annot identiques)', '',
              'Pourcentage positif : réduction de l’erreur. Pourcentage négatif : augmentation de l’erreur. Aucun de ces pourcentages n’est une accuracy.', '',
              '| Run | Réduction relative de MCRMSE |', '|---|---:|']
    for _,r in df.dropna(subset=['error_reduction_vs_bigru_pct']).iterrows():
        lines.append(f"| {r['run']} | {r['error_reduction_vs_bigru_pct']:+.2f} % |")
    lines+=['','Les différences entre architectures sont comparées au niveau annot. Les différences annot → graph mesurent l’apport des interactions explicites à caractéristiques par nœud identiques. Les tailles des modèles diffèrent : cette comparaison ne fixe pas exactement le nombre de paramètres.', '',
            'GCN graph additionne squelette, paires et probabilités BPPS avant normalisation. Transformer graph ajoute un biais BPPS appris par tête. Les annotations et probabilités proviennent de prédictions de structure secondaire. KFold ne garantit pas la séparation de séquences similaires ; le faible effectif, une seule seed et la sélection sur les folds limitent les conclusions. Le budget est limité à 40 époques : les modèles ne sont pas nécessairement convergés. Les scores locaux ne sont pas directement comparables au leaderboard privé Kaggle.']
    for prefix in ['pilot_','cv_']:
        d=df[df.run.str.startswith(prefix)].set_index('run')
        if d.empty: continue
        fig,ax=plt.subplots(figsize=(9,4)); d.mcrmse.plot.bar(ax=ax,color='#398789'); ax.set(ylabel='MCRMSE OOF (plus faible = meilleur)',title='Fold 0' if prefix=='pilot_' else 'Validation croisée'); fig.tight_layout(); fig.savefig(out/f'{prefix}mcrmse.png',dpi=160); plt.close(fig)
        fig,ax=plt.subplots(figsize=(10,4)); d[['reactivity','deg_Mg_pH10','deg_Mg_50C']].plot.bar(ax=ax); ax.set(ylabel='RMSE par cible'); fig.tight_layout(); fig.savefig(out/f'{prefix}targets.png',dpi=160); plt.close(fig)
        ablation=d.pivot(index='model',columns='level',values='mcrmse')
        ablation=ablation.reindex(columns=[c for c in ['seq','annot','graph'] if c in ablation.columns])
        fig,ax=plt.subplots(figsize=(7,4)); ablation.plot.bar(ax=ax); ax.set(ylabel='MCRMSE',xlabel='Architecture',title="Contribution de l’information structurale"); fig.tight_layout(); fig.savefig(out/f'{prefix}structure.png',dpi=160); plt.close(fig)
        lines += ['', '## '+('Pilotes (fold 0)' if prefix=='pilot_' else 'Évaluation à plusieurs folds'), '']
        for model, values in ablation.iterrows():
            for a,b in [('seq','annot'),('annot','graph')]:
                if a in values and b in values and pd.notna(values[a]) and pd.notna(values[b]):
                    delta=values[a]-values[b]
                    lines.append(f'- {model}, {a} → {b} : variation de MCRMSE {values[b]-values[a]:+.5f} ({"amélioration" if delta>0 else "dégradation"}). Observation descriptive, sans conclusion de significativité.')
    (out/'methodologie_resultats.md').write_text('\n'.join(lines)+'\n')
    print(df[['run','folds','mcrmse','seconds']].to_string(index=False))
