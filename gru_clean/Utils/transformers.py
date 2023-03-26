import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class StaticPositionEmbedding(tf.keras.layers.Layer):
    '''Static positional embedding with trigonometric function'''

    def __init__(self, max_len=160):
        super(StaticPositionEmbedding, self).__init__()
        self.max_len = max_len

    def build(self, input_shape):
        super(StaticPositionEmbedding, self).build(input_shape)

    def call(self, x, masking=True):
        E = x.get_shape().as_list()[-1]  # static
        batch_size, seq_length = tf.shape(x)[0], tf.shape(x)[1]  # dynamic

        pos_idx = tf.tile(tf.expand_dims(tf.range(seq_length), 0),
                          [batch_size, 1])  # => batch_size * seq_length
        pos_encode = np.array(
            [[pos / np.power(10000, (i - i % 2) / E)
              for i in range(E)]
             for pos in range(self.max_len)])

        pos_encode[:, 0::2] = np.sin(pos_encode[:, 0::2])
        pos_encode[:, 1::2] = np.cos(pos_encode[:, 1::2])
        pos_encode = tf.convert_to_tensor(pos_encode, tf.float32)  # (maxlen, E)

        outputs = tf.nn.embedding_lookup(pos_encode, pos_idx)
        if masking:
            outputs = tf.where(tf.equal(x, 0), x, outputs)
        return tf.cast(outputs, tf.float32)

    def get_config(self):
        config = {"max_len": self.max_len}
        base_config = super(StaticPositionEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DynamicPositionEmbedding(tf.keras.layers.Layer):
    '''Dynamic positional embedding'''

    def __init__(self,
                 max_length=500,
                 initializer="glorot_uniform",
                 seq_axis=1,
                 **kwargs):

        super().__init__(**kwargs)
        if max_length is None:
            raise ValueError("`max_length` must be an Integer, not `None`.")
        self._max_length = max_length
        self._initializer = tf.keras.initializers.get(initializer)
        self._seq_axis = seq_axis

    def get_config(self):
        config = {
            "max_length": self._max_length,
            "initializer": tf.keras.initializers.serialize(self._initializer),
            "seq_axis": self._seq_axis,
        }
        base_config = super(DynamicPositionEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        dimension_list = input_shape.as_list()
        width = dimension_list[-1]
        weight_sequence_length = self._max_length

        self._position_embeddings = self.add_weight(
            "embeddings",
            shape=[weight_sequence_length, width],
            initializer=self._initializer)

        super().build(input_shape)

    def call(self, inputs):
        input_shape = tf.shape(inputs)
        actual_seq_len = input_shape[self._seq_axis]
        position_embeddings = self._position_embeddings[:actual_seq_len, :]
        new_shape = [1 for _ in inputs.get_shape().as_list()]
        new_shape[self._seq_axis] = actual_seq_len
        new_shape[-1] = position_embeddings.get_shape().as_list()[-1]
        position_embeddings = tf.reshape(position_embeddings, new_shape)
        return tf.broadcast_to(position_embeddings, input_shape)


def transformer_encoder(inputs, unit_head, n_heads, n_bottleneck_units, dropout=0):
    # Norm & multi-head attention
    x = layers.LayerNormalization(epsilon=1e-7)(inputs)
    x = layers.MultiHeadAttention(key_dim=unit_head,
                                  num_heads=n_heads,
                                  dropout=dropout)(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # FC
    x = layers.LayerNormalization(epsilon=1e-7)(res)
    # Use Conv1D instead of Dense
    x = layers.Conv1D(filters=n_bottleneck_units, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res


def build_transformer_base(input_shape,
                           position_embedding='dynamic',
                           unit_head=50,
                           n_heads=2,
                           n_bottleneck_units=64,
                           num_transformer_blocks=2,
                           mlp_units=[32],
                           dropout=0.25,
                           mlp_dropout=0.5):
    '''Transformer with optional positional(temporal) embedding'''

    inputs = keras.Input(shape=input_shape)
    x = inputs
    x = layers.Dense(input_shape[-1])(x)

    if position_embedding == 'static':
        pos_emb = StaticPositionEmbedding()(inputs)
        pos_emb = layers.Dense(input_shape[-1])(pos_emb)
        x = tf.add(x, pos_emb)
    elif position_embedding == 'dynamic':
        pos_emb = DynamicPositionEmbedding()(inputs)
        x = tf.add(x, pos_emb)

    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, unit_head, n_heads, n_bottleneck_units, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_last")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="swish")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(1)(x)
    return keras.Model(inputs, outputs)