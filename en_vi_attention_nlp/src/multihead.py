import tensorflow as tf
import tensorflow.keras.backend as K
from keras.layers import Dense, Lambda, TimeDistributed, \
    Add, Dropout, Concatenate, Activation, Layer
from keras.initializers import Ones, Zeros
from kulc.attention import ScaledDotProductAttention

class MultiHeadAttention(object):
    # mode 0 - big matrices, faster; mode 1 - more clear implementation
    def __init__(self, h, d_k, d_v, d_model, dropout=0.1, return_attention=False, mode=0):
        self._mode = mode
        self._n_head = h
        self._d_k = d_k
        self._d_v = d_v
        self._d_model = d_model
        self._dropout = dropout
        self._return_attention = return_attention
        self._qs_layers = []
        self._ks_layers = []
        self._vs_layers = []
        for _ in range(self._n_head):
            self._qs_layers.append(TimeDistributed(Dense(d_k, use_bias=False)))
            self._ks_layers.append(TimeDistributed(Dense(d_k, use_bias=False)))
            self._vs_layers.append(TimeDistributed(Dense(d_v, use_bias=False)))
        self._attention = ScaledDotProductAttention(
            return_attention=return_attention)
        self._w_o = TimeDistributed(Dense(self._d_model))

    def __call__(self, x, mask=None):
        q, k, v = x
        outputs = []
        attentions = []
        for i in range(self._n_head):
            qi = self._qs_layers[i](q)
            ki = self._ks_layers[i](k)
            vi = self._vs_layers[i](v)

            if self._return_attention:
                output, attention = self._attention([qi, ki, vi], mask=mask)
                outputs.append(output)
                attentions.append(attention)
            else:
                output = self._attention([qi, ki, vi], mask=mask)
                outputs.append(output)

        concatenated_outputs = Concatenate()(outputs)
        output = self._w_o(concatenated_outputs)

        if self._return_attention:
            attention = Concatenate()(attentions)
            # print("attention", attention, attention.shape)

        if self._return_attention:
            return [output, attention]
        else:
            return output


class LayerNormalization(Layer):
    def __init__(self, eps=1e-6, **kwargs):
        self.eps = eps
        super(LayerNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.gamma = self.add_weight(
            name='gamma', shape=input_shape[-1:], initializer=Ones(), trainable=True)
        self.beta = self.add_weight(
            name='beta', shape=input_shape[-1:], initializer=Zeros(), trainable=True)
        super(LayerNormalization, self).build(input_shape)

    def call(self, x, **kwargs):
        mean = K.mean(x, axis=-1, keepdims=True)
        std = K.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

    def compute_output_shape(self, input_shape):
        return input_shape
