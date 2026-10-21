"""Vérification indépendante des résultats CV à partir des OOF et des CSV."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src import config as C
from experiments.run_comparison import MATRIX


def score(y,p):
    return float(np.sqrt(np.mean((y[...,C.SCORED_IDX]-p[...,C.SCORED_IDX])**2,axis=(0,1))).mean())


def main():
    with np.load(C.PROCESSED_DIR/'train.npz',allow_pickle=False) as z:
        y=z['targets'];clean=z['sn_filter']==1;ids=z['ids']
        assignments={s:z['fold_'+s] for s in ['kfold','group']}
    for s,f in assignments.items():
        csv=pd.read_csv(C.FOLDS_DIR/f'folds_{s}.csv').set_index('id').loc[ids,'fold'].to_numpy()
        np.testing.assert_array_equal(f,csv)
    report=[]
    protocol=None
    for model,level,params in MATRIX:
        name=f'cv_{model}_{level}';path=C.OUTPUT_DIR/name
        data=json.loads((path/'scores.json').read_text())
        assert data['args']['model']==model and data['args']['level']==level,name
        assert json.loads(data['args']['params'])==params,name
        current={k:data['args'][k] for k in ['folds','optimizer','lr','batch_size','epochs','patience','seed','all_samples','skip_test']}
        if protocol is None: protocol=current
        assert current==protocol,f'protocole différent : {name}'
        assert set(data['fold_mcrmse'])==set(map(str,range(C.N_FOLDS))),name
        fold=assignments[data['args']['folds']];p=np.load(path/'oof.npy')
        assert p.shape==y.shape and np.isfinite(p).all(),name
        np.testing.assert_allclose(score(y[clean],p[clean]),data['cv_mcrmse'],atol=1e-7,rtol=0)
        values=[]
        for k in range(C.N_FOLDS):
            mask=clean&(fold==k);v=score(y[mask],p[mask]);values.append(v)
            np.testing.assert_allclose(v,data['fold_mcrmse'][str(k)],atol=1e-7,rtol=0)
            h=pd.read_csv(path/f'history_fold{k}.csv')
            np.testing.assert_allclose(v,h.val_mcrmse.min(),atol=1e-7,rtol=0)
            assert (path/f'fold{k}.weights.h5').is_file(),name
        np.testing.assert_allclose([np.mean(values),np.std(values,ddof=1)],[data['fold_mean'],data['fold_std']],atol=1e-7,rtol=0)
        source=data.get('reused_pilot')
        if source:
            original=C.OUTPUT_DIR/source
            np.testing.assert_array_equal(p[fold==0],np.load(original/'oof.npy')[fold==0])
            for filename in ['fold0.weights.h5','history_fold0.csv']:
                assert hashlib.sha256((path/filename).read_bytes()).digest()==hashlib.sha256((original/filename).read_bytes()).digest()
        report.append(dict(run=name,folds=5,mcrmse=data['cv_mcrmse'],reused_pilot=source))
        print(name,'verified',flush=True)
    result=dict(runs=report,folds_verified=len(report)*C.N_FOLDS,
                fold_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in C.FOLDS_DIR.glob('*.csv')})
    (C.OUTPUT_DIR/'comparison/verification.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
