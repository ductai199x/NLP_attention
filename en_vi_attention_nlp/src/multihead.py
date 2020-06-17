import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense, Lambda, TimeDistributed, \
    Add, Dropout, Concatenate, Activation, Layer
from tensorflow.keras.initializers import Ones, Zeros


class ScaledDotProductAttention():
    def __init__(self, attn_dropout=0.1):
        self._dropout = Dropout(attn_dropout)

    def __call__(self, q, k, v, mask):  # mask_k or mask_qk
        temper = tf.sqrt(tf.cast(tf.shape(k)[-1], dtype='float32'))
        attn = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[2, 2]) / temper)([q, k])  # shape=(batch, q, k)
        if mask is not None:
            mmask = Lambda(lambda x: (-1e+9) * (1. - K.cast(x, 'float32')))(mask)
            attn = Add()([attn, mmask])
        attn = Activation('softmax')(attn)
        attn = self._dropout(attn)
        output = Lambda(lambda x: K.batch_dot(x[0], x[1]))([attn, v])
        return output, attn


class MultiHeadAttention(object):
    # mode 0 - big matrices, faster; mode 1 - more clear implementation
    def __init__(self, n_head, d_k, d_v, d_model, dropout=0.1, return_attention=False, mode=0):
        self._mode = mode
        self._n_head = n_head
        self._d_k = d_k
        self._d_v = d_v
        self._d_model = d_model
        self._dropout = dropout
        self._return_attention = return_attention
        if self._mode == 0:
            self._qs_layer = Dense(n_head * d_k, use_bias=False)
            self._ks_layer = Dense(n_head * d_k, use_bias=False)
            self._vs_layer = Dense(n_head * d_v, use_bias=False)
        elif self._mode == 1:
            self._qs_layers = []
            self._ks_layers = []
            self._vs_layers = []
            for _ in range(n_head):
                self._qs_layers.append(TimeDistributed(Dense(d_k, use_bias=False)))
                self._ks_layers.append(TimeDistributed(Dense(d_k, use_bias=False)))
                self._vs_layers.append(TimeDistributed(Dense(d_v, use_bias=False)))
        self._attention = ScaledDotProductAttention()
        self._w_o = TimeDistributed(Dense(d_model))

    def __call__(self, q, k, v, mask=None):
        d_k, d_v = self._d_k, self._d_v
        n_head = self._n_head

        head = None
        attn = None

        if self._mode == 0:
            qs = self._qs_layer(q)  # [batch_size, len_q, n_head*d_k]
            ks = self._ks_layer(k)
            vs = self._vs_layer(v)

            def reshape1(x):
                s = tf.shape(x)  # [batch_size, len_q, n_head * d_k]
                x = tf.reshape(x, [s[0], s[1], n_head, s[2] // n_head])
                x = tf.transpose(x, [2, 0, 1, 3])
                x = tf.reshape(x, [-1, s[1], s[2] // n_head])  # [n_head * batch_size, len_q, d_k]
                return x

            qs = Lambda(reshape1)(qs)
            ks = Lambda(reshape1)(ks)
            vs = Lambda(reshape1)(vs)

            if mask is not None:
                mask = Lambda(lambda x: K.repeat_elements(x, n_head, 0))(mask)
            head, attn = self._attention(qs, ks, vs, mask=mask)

            def reshape2(x):
                s = tf.shape(x)  # [n_head * batch_size, len_v, d_v]
                x = tf.reshape(x, [n_head, -1, s[1], s[2]])
                x = tf.transpose(x, [1, 2, 0, 3])
                x = tf.reshape(x, [-1, s[1], n_head * d_v])  # [batch_size, len_v, n_head * d_v]
                return x

            head = Lambda(reshape2)(head)
        elif self._mode == 1:
            heads = []
            attns = []
            for i in range(n_head):
                qs = self._qs_layers[i](q)
                ks = self._ks_layers[i](k)
                vs = self._vs_layers[i](v)
            head, attn = self._attention(qs, ks, vs, mask)
            heads.append(head)
            attns.append(attn)
            head = Concatenate()(heads) if n_head > 1 else heads[0]
            attn = Concatenate()(attns) if n_head > 1 else attns[0]

        outputs = self._w_o(head)
        outputs = Dropout(self._dropout)(outputs)

        if self._return_attention:
            return outputs, attn
        else:
            return outputs


class LayerNormalization(Layer):
    def __init__(self, eps=1e-6, **kwargs):
        self.eps = eps
        super(LayerNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:], initializer=Ones(), trainable=True)
        self.beta = self.add_weight(name='beta', shape=input_shape[-1:], initializer=Zeros(), trainable=True)
        super(LayerNormalization, self).build(input_shape)

    def call(self, x, **kwargs):
        mean = K.mean(x, axis=-1, keepdims=True)
        std = K.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

    def compute_output_shape(self, input_shape):
        return input_shape
