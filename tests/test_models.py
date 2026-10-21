"""Vérifications numériques : longueurs, gradients, niveaux et checkpoints."""
import os
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '1')
import tempfile
import unittest
from pathlib import Path
import keras
import numpy as np
import tensorflow as tf
from src.models import MODELS
from src.models.graph import RnaGraph
from src.evaluation.metrics import mcrmse_loss


class ModelTests(unittest.TestCase):
    def inputs(self, length):
        rng = np.random.default_rng(42)
        tokens = np.stack([rng.integers(0, 4, (2, length)), rng.integers(4, 7, (2, length)), rng.integers(7, 14, (2, length))], axis=-1).astype('int32')
        b = rng.uniform(0, .1, (2, length, length)).astype('float32')
        return dict(tokens=tokens, feats=rng.random((2, length, 4)).astype('float32'), bpps=(b+b.transpose(0,2,1))/2, pairs=np.full((2,length), -1, dtype='int32'))

    def test_models(self):
        for name, level in [('gru','seq'),('gru','annot'),('conv_gru','annot'),('attention_gru','annot'),('transformer','annot'),('transformer','graph'),('gcn','annot'),('gcn','graph'),('gat','annot'),('gat','graph')]:
            with self.subTest(model=name, level=level):
                keras.backend.clear_session()
                keras.utils.set_random_seed(42)
                params = dict(hidden=16) if name in ('gru','conv_gru','attention_gru') else (dict(d_model=32, num_heads=4, n_layers=2, ff_dim=64) if name=='transformer' else dict(units=32,n_layers=2))
                m=MODELS[name](4,level=level,**params)
                keys=[t.name for t in m.inputs]
                x={k:v for k,v in self.inputs(107).items() if k in keys}
                if name=='gat' and level=='graph':
                    x['bpps'][:,0,-1]=x['bpps'][:,-1,0]=.5
                with tf.GradientTape() as tape:
                    y=m(x,training=True)
                    loss=mcrmse_loss(tf.zeros((2,68,5)),y)
                grads=tape.gradient(loss,m.trainable_weights)
                present=[g for g in grads if g is not None]
                self.assertTrue(present)
                self.assertTrue(all(bool(tf.reduce_all(tf.math.is_finite(g.values if isinstance(g,tf.IndexedSlices) else g))) for g in present))
                keras.optimizers.AdamW().apply_gradients((g,w) for g,w in zip(grads,m.trainable_weights) if g is not None)
                pred=m(x,training=False).numpy()
                self.assertEqual(pred.shape,(2,107,5))
                self.assertTrue(np.isfinite(pred).all())
                long={k:v for k,v in self.inputs(130).items() if k in keys}
                self.assertEqual(tuple(m(long).shape),(2,130,5))
                if level=='seq':
                    changed={**x,'tokens':x['tokens'].copy()}; changed['tokens'][:,:,1:]=4
                    np.testing.assert_allclose(pred,m(changed,training=False),atol=1e-6)
                if level=='graph':
                    changed={**x,'bpps':np.zeros_like(x['bpps'])}
                    self.assertGreater(float(np.max(np.abs(pred-m(changed,training=False).numpy()))),1e-7)
                    if name=='gat':
                        same_edges={**x,'bpps':x['bpps'].copy()}
                        same_edges['bpps'][:,0,-1]=same_edges['bpps'][:,-1,0]=.8
                        np.testing.assert_allclose(pred,m(same_edges,training=False),atol=1e-6)
                with tempfile.TemporaryDirectory() as d:
                    p=Path(d)/'test.weights.h5'; m.save_weights(p)
                    restored=MODELS[name](4,level=level,**params); restored.load_weights(p)
                    np.testing.assert_allclose(pred,restored(x,training=False),atol=1e-6)

    def test_mask(self):
        y=tf.zeros((2,68,5)); a=np.zeros((2,107,5),dtype='float32'); b=a.copy(); b[:,68:]=100
        np.testing.assert_allclose(mcrmse_loss(y,a),mcrmse_loss(y,b))

    def test_graph(self):
        x=tf.zeros((1,4,2)); pairs=tf.constant([[3,-1,-1,0]]); b=tf.zeros((1,4,4))
        a=RnaGraph()(x,pairs,b).numpy()
        self.assertEqual(a.shape,(1,3,4,4))
        np.testing.assert_array_equal(a[0,0],[[0,1,0,0],[1,0,1,0],[0,1,0,1],[0,0,1,0]])
        self.assertEqual(a[0,1,0,3],1)
        self.assertEqual(a[0,1,3,0],1)


if __name__=='__main__':
    unittest.main()
