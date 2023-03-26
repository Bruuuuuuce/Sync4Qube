import os

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras.layers import (
    GRU,
    LSTM,
    Conv1D,
    BatchNormalization,
    LayerNormalization,
    Attention,
    MultiHeadAttention,
    Concatenate,
    GlobalAveragePooling1D,
    Dense,
    Dropout,
    Embedding,
    Flatten,
    Input,
    LeakyReLU,
    Reshape,
)
from tensorflow.keras.models import Sequential

import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import *
from tensorflow.keras.initializers import *
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.python.keras.layers import Layer


def gru_block(inputs, units_gru, num_gru_layers, dropout=0.0):

    x = LayerNormalization(epsilon=1e-6)(inputs)
    # x = inputs

    for _ in range(num_gru_layers):
        x = GRU(units_gru,
                return_sequences=True,
                kernel_regularizer=keras.regularizers.L1L2(l1=1e-6, l2=1e-6),
                dropout=dropout)(x)

    outputs = x
    return outputs


def self_attention_block(inputs, unit_head, n_heads, n_features, dropout=0.0):

    # Norm & Attention
    x = LayerNormalization(epsilon=1e-6)(inputs)
    # x = inputs
    x = Attention()([x, x])
    # x = MultiHeadAttention(key_dim=unit_head,
    #                        num_heads=n_heads,
    #                        dropout=dropout)(x, x)
    x = Dropout(dropout)(x)
    res = x + inputs

    # FC
    x = LayerNormalization(epsilon=1e-6)(res)
    # x = res
    # Use Conv1D instead of Dense
    x = Conv1D(filters=n_features, kernel_size=1, activation="swish")(x)
    x = Dropout(dropout)(x)
    x = Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res


def build_gru_selfattn(input_shape=(160, 80),
                       num_gru_layers=1,
                       mlp_units=[32],
                       n_attention_blocks=1,
                       unit_head=80,
                       n_heads=1,
                       n_features=64,
                       dropout=0.3,
                       mlp_dropout=0.5):

    inputs = keras.Input(shape=input_shape)
    x = inputs

    x = Dense(input_shape[-1], activation='relu')(x)
    x = gru_block(x,
                  units_gru=input_shape[-1],
                  num_gru_layers=num_gru_layers,
                  dropout=dropout)

    for _ in range(n_attention_blocks):
        x = self_attention_block(x, unit_head, n_heads, n_features, dropout)

    # x = GlobalAveragePooling1D(data_format="channels_first")(x)
    x = GlobalAveragePooling1D(data_format="channels_last")(x)
    for dim in mlp_units:
        x = Dense(dim, activation='swish')(x)
        x = Dropout(mlp_dropout)(x)
    outputs = Dense(1)(x)
    return keras.Model(inputs, outputs)


# =============================================================================================
# =============================================================================================


class myAttention(Layer):

    def __init__(self,
                 step_dim,
                 W_regularizer=None,
                 b_regularizer=None,
                 W_constraint=None,
                 b_constraint=None,
                 bias=True,
                 **kwargs):
        self.supports_masking = True
        self.initializer = tf.keras.initializers.get('glorot_uniform')
        # W_regularizer: 权重正则化
        # b_regularizer: 偏置正则化
        self.W_regularizer = tf.keras.regularizers.get(W_regularizer)
        self.b_regularizer = tf.keras.regularizers.get(b_regularizer)
        # W_constraint: 权重约束项
        # b_constraint: 偏置约束项
        self.W_constraint = tf.keras.constraints.get(W_constraint)
        self.b_constraint = tf.keras.constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(myAttention, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'step_dim': self.step_dim,
            'W_regularizer': self.W_regularizer,
            'b_regularizer': self.b_regularizer,
            'W_constraint': self.W_constraint,
            'b_constraint': self.b_constraint,
            'bias': self.bias
        })
        return config

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.features_dim = input_shape[-1]

        self.W = self.add_weight(shape=(self.features_dim,),
                                 initializer=self.initializer,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight(shape=(input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(
            K.dot(K.reshape(x, (-1, features_dim)),
                  K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)
        '''
        将张量转换到不同的 dtype 并返回
        '''
        if mask is not None:
            a *= K.cast(mask, K.floatx())
        '''
        返回浮点
        '''
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a

        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.features_dim


def build_gru_attn(input_shape=(160, 80),
                   num_gru_layers=1,
                   mlp_units=[32],
                   n_attention_blocks=1,
                   dropout=0.3,
                   mlp_dropout=0.5,
                   **kwargs):

    inputs = Input(shape=input_shape)
    x = inputs

    x = Dense(input_shape[-1], activation='relu')(x)
    x = gru_block(x,
                  units_gru=input_shape[-1],
                  num_gru_layers=num_gru_layers,
                  dropout=dropout)
    x = Dropout(dropout)(x)
    x = LayerNormalization(epsilon=1e-6)(x)

    x = myAttention(input_shape[0])(x)

    for dim in mlp_units:
        x = Dense(dim, activation='swish')(x)
        x = Dropout(mlp_dropout)(x)

    outputs = Dense(1)(x)
    return keras.Model(inputs, outputs)
