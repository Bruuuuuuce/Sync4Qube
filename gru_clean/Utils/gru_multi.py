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


def build_gru_multi(units_gru=50,
                    num_gru_layers=2,
                    mlp_units=[20],
                    input_shape=(160, 11),
                    dropout=0.3,
                    mlp_dropout=0.5,
                    **kwargs):
    model = Sequential()
    model.add(
        GRU(units_gru,
            return_sequences=True,
            input_shape=input_shape,
            kernel_regularizer=keras.regularizers.L1L2(l1=1e-6, l2=1e-6),
            dropout=dropout))
    if num_gru_layers > 2:
        for _ in range(num_gru_layers - 2):
            model.add(
                GRU(units_gru,
                    return_sequences=True,
                    kernel_regularizer=keras.regularizers.L1L2(l1=1e-6,
                                                               l2=1e-6),
                    dropout=dropout))
    model.add(
        GRU(units_gru,
            return_sequences=False,
            kernel_regularizer=keras.regularizers.L1L2(l1=1e-6, l2=1e-6),
            dropout=dropout))
    model.add(Dropout(mlp_dropout))
    for dim in mlp_units:
        model.add(
            Dense(dim,
                  activation=keras.activations.swish,
                  kernel_regularizer=keras.regularizers.L1L2(l1=1e-6, l2=1e-6)))
        model.add(Dropout(mlp_dropout))
    model.add(Dense(1))
    return model
