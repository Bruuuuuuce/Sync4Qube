import os

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras.layers import (
    GRU,
    LSTM,
    BatchNormalization,
    Concatenate,
    Dense,
    Dropout,
    Embedding,
    Flatten,
    Input,
    LeakyReLU,
    Reshape,
)
from tensorflow.keras.models import Sequential


def build_gru_base(units_gru=50, units_dense=20, input_shape=(160, 11), **kwargs):
    model = Sequential()
    model.add(
        GRU(units_gru,
            return_sequences=False,
            input_shape=input_shape,
            kernel_regularizer=keras.regularizers.L1L2(l1=1e-6, l2=1e-6),
            dropout=0.3))
    model.add(Dropout(0.3))
    model.add(
        Dense(units_dense,
              activation=keras.activations.swish,
              kernel_regularizer=keras.regularizers.L1L2(l1=1e-6, l2=1e-6)))
    model.add(Dense(1))
    return model
