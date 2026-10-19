"""Modèle 2 : Transformer Encoder ; au niveau "graph", l'attention est biaisée par la BPPS.

scores = Q K^T / sqrt(d) + w_h * BPPS     (un poids w_h appris par tête)

La BPPS joue le rôle de la « pair representation » d'AlphaFold2 : elle indique
au modèle quelles positions interagissent en 3D.
"""
import keras
from keras import layers, ops

from .common import embed, model_inputs, output_head, used_inputs


class SinusoidalPositionEncoding(layers.Layer):
    """Encodage positionnel de « Attention is all you need » (N = 10000)."""

    def call(self, x):
        L, d = ops.shape(x)[1], x.shape[-1]
        pos = ops.cast(ops.arange(L), "float32")[:, None]
        i = ops.cast(ops.arange(d // 2), "float32")[None, :]
        theta = pos / ops.power(10000.0, 2 * i / d)
        pe = ops.reshape(ops.stack([ops.sin(theta), ops.cos(theta)], axis=-1), (L, d))
        return x + pe[None]


class BppsAttention(layers.Layer):
    def __init__(self, d_model, num_heads, use_bpps=True, **kwargs):
        super().__init__(**kwargs)
        self.d_model, self.h, self.use_bpps = d_model, num_heads, use_bpps
        self.dk = d_model // num_heads
        self.qkv = layers.Dense(3 * d_model)
        self.out = layers.Dense(d_model)

    def build(self, input_shape):
        self.bpps_weight = self.add_weight(shape=(self.h,), initializer="ones", name="bpps_weight")

    def call(self, x, bpps=None):
        B, L = ops.shape(x)[0], ops.shape(x)[1]
        qkv = ops.reshape(self.qkv(x), (B, L, 3, self.h, self.dk))
        q, k, v = [ops.transpose(qkv[:, :, i], (0, 2, 1, 3)) for i in range(3)]  # (B, h, L, dk)
        scores = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2))) / (self.dk ** 0.5)
        if self.use_bpps:
            scores = scores + self.bpps_weight[None, :, None, None] * bpps[:, None]
        attn = ops.softmax(scores, axis=-1)
        y = ops.transpose(ops.matmul(attn, v), (0, 2, 1, 3))
        return self.out(ops.reshape(y, (B, L, self.d_model)))


def build_model(n_feats, level="graph", embed_dim=32, d_model=128, num_heads=8, n_layers=4,
                ff_dim=256, dropout=0.1):
    """level = "graph" : la BPPS biaise l'attention ; sinon self-attention classique."""
    inputs = model_inputs(n_feats)
    use_bpps = level == "graph"
    x = layers.Dense(d_model)(embed(inputs, embed_dim, level))
    x = SinusoidalPositionEncoding()(x)

    for _ in range(n_layers):
        a = BppsAttention(d_model, num_heads, use_bpps)(x, inputs["bpps"] if use_bpps else None)
        x = layers.LayerNormalization()(x + layers.Dropout(dropout)(a))
        f = layers.Dense(ff_dim, activation="gelu")(x)
        f = layers.Dense(d_model)(f)
        x = layers.LayerNormalization()(x + layers.Dropout(dropout)(f))

    return keras.Model(used_inputs(inputs, level), output_head(x), name=f"transformer_{level}")
