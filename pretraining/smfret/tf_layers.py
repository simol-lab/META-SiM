"""Custom tensorflow layers."""

import tensorflow as tf
import tensorflow.keras as keras


@keras.utils.register_keras_serializable('Attention')
class Attention(keras.layers.Layer):
    def __init__(self, inner_dim=32, head_dim=2):
        super().__init__()
        self.inner_dim = inner_dim
        self.head_dim = head_dim
        
    def get_config(self):
        config = {
            "inner_dim": self.inner_dim,
            "head_dim": self.head_dim,
        }
        return config
    
    def build(self, input_shape):
        input_dim = int(input_shape[-1])
        self.q = self.add_weight("Q", shape=[self.head_dim, input_dim, self.inner_dim], initializer=tf.keras.initializers.GlorotUniform())
        self.k = self.add_weight("K", shape=[self.head_dim, input_dim, self.inner_dim], initializer=tf.keras.initializers.GlorotUniform())
        self.v = self.add_weight("V", shape=[self.head_dim, input_dim, self.inner_dim], initializer=tf.keras.initializers.GlorotUniform())
        self.linear = self.add_weight("linear", shape=[self.head_dim, self.inner_dim, input_dim])
    
    def get_weights(self, inputs, instruct_token):
        d = tf.cast(self.inner_dim, inputs.dtype)
        Q = tf.einsum('bti,hij->bhtj', inputs, self.q) 
        K = tf.einsum('bti,hij->bhtj', inputs, self.k)
        logits = tf.einsum('bhti,bhri->bhtr', Q, K) / tf.sqrt(d)  # same as paper, but do not understand
        if instruct_token is not None:
            one = tf.ones(shape=(1, ), dtype=Q.dtype)
            logits += tf.einsum('t,i,bhri->bhtr', one, instruct_token, K)
        weights = tf.nn.softmax(tf.einsum('bhti,bhri->bhtr', Q, K), axis=-1)
        return weights
    
    @tf.function
    def call(self, inputs, instruct_token=None):
        weights = self.get_weights(inputs, instruct_token)
        V = tf.einsum('bti,hij->bhtj', inputs, self.v)
        weighted_output = tf.einsum('bhri,bhtr->bhti', V, weights)
        output = inputs + tf.einsum('bhti,hij->btj', weighted_output, self.linear)
        normalized_output = tf.math.l2_normalize(output, axis=-1)
        return normalized_output


@keras.utils.register_keras_serializable('Attention')
class Conv(keras.layers.Layer):
    def __init__(self, width=8, inner_dim=32):
        super().__init__()
        self.inner_dim = inner_dim
        self.width = width
        self.stride = width // 2
        
    def get_config(self):
        config = {
            "inner_dim": self.inner_dim,
            "width": self.width,
        }
        return config
    
    def build(self, input_shape):
        input_dim = int(input_shape[-1])
        self.filter = self.add_weight("weight", shape=[self.width, input_dim, self.inner_dim], initializer=tf.keras.initializers.GlorotUniform())
        self.bias = self.add_weight("bias", shape=[1, 1, self.inner_dim], initializer=tf.zeros_initializer())
    
    @tf.function
    def call(self, inputs):
        output = tf.nn.conv1d(
            input=inputs,
            filters=self.filter,
            stride=self.stride,
            padding='SAME',
        )
        output -= self.bias
        return output
    
    def reconstruct(self, inputs):
        output = tf.repeat(inputs, self.stride, axis=-2)
        return output


@keras.utils.register_keras_serializable('Attention')
class Reconstructor(keras.layers.Layer):
    def __init__(self, width=8):
        super().__init__()
        self.width = width
        self.stride = width // 2

    def get_config(self):
        config = {
            "width": self.width,
        }
        return config
    
    def build(self, input_shape):
        input_dim = int(input_shape[-1])
        self.input_dim = input_dim
        self.remap = self.add_weight("remap", shape=[input_dim, self.stride, input_dim], initializer=tf.zeros_initializer())
    
    @tf.function
    def call(self, inputs):
        base = tf.repeat(inputs, self.stride, axis=-2)
        
        z = tf.einsum('iTh,htz->iTtz', inputs, self.remap)
        shape = tf.shape(z)
        new_shape = (shape[0], shape[1] * shape[2], shape[3])
        z = tf.reshape(z, new_shape)
                
        output = base + z
        output = tf.math.l2_normalize(output, axis=-1)
        return z


@keras.utils.register_keras_serializable('Attention')
class Summary(keras.layers.Layer):
    def __init__(self, inner_dim=16, outer_dim=16, head_dim=8):
        super().__init__()
        self.inner_dim = inner_dim
        self.outer_dim = outer_dim
        self.head_dim = head_dim
        
    def get_config(self):
        config = {
            "inner_dim": self.inner_dim,
            "outer_dim": self.outer_dim,
            "head_dim": self.head_dim,
        }
        return config
    
    def build(self, input_shape):
        input_dim = int(input_shape[-1])
        self.k = self.add_weight("K", shape=[self.head_dim, input_dim], initializer=tf.keras.initializers.GlorotUniform())
        self.v = self.add_weight("V", shape=[self.head_dim, input_dim, self.outer_dim], initializer=tf.keras.initializers.GlorotUniform())
        self.b = self.add_weight("B", shape=(1, self.outer_dim), initializer=tf.zeros_initializer())
    
    @tf.function
    def call(self, inputs):
        k = tf.math.l2_normalize(self.k, axis=-1)
        v = self.v
        inputs = tf.math.l2_normalize(inputs, axis=-1)
        w = tf.einsum("hi,bti->bht", k, inputs) / tf.sqrt(tf.cast(self.inner_dim, inputs.dtype))
        w = tf.nn.softmax(w, axis=-1)
        logits = tf.einsum("bht,hij,bti->bj", w, v, inputs) + self.b
        return logits


@keras.utils.register_keras_serializable('Attention')
class PrependTaskToken(tf.keras.layers.Layer):
    """Keras layer for prepending a learnable token to the sequence."""
    def __init__(self, token_dim, dtype=tf.float32):
        super().__init__()
        self.token_dim = token_dim
    
    def build(self, input_shape):
        self.token = self.add_weight("token", shape=[1, 1, self.token_dim], initializer=tf.keras.initializers.GlorotUniform())
        
    def get_config(self):
        config = {
            "token_dim": self.token_dim,
        }
        return config
    
    def call(self, seq):
        batch_size = tf.shape(seq)[0]
        token_tiled = tf.repeat(self.token, batch_size, axis=0)
        return tf.concat([token_tiled, seq], axis=1)


@keras.utils.register_keras_serializable('Attention')    
class Embedding(keras.layers.Layer):
    """Full e2e embedding layer for trace sequence analysis."""
    def __init__(self, conv, transformer, prepend_task_token, reconstructor=None, framewise=False):
        super().__init__()
        self.conv = conv
        self.transformer = transformer
        self.prepend_task_token = prepend_task_token
        self.framewise = framewise
        if reconstructor is not None:
            self.reconstructor = reconstructor
        else:
            self.reconstructor = conv.reconstruct
        
    def get_config(self):
        config = {
            "conv": self.conv,
            "transformer": self.transformer,
            "prepend_task_token": self.prepend_task_token,
            "reconstructor": self.reconstructor,
            "framewise": self.framewise,
        }
        return config
    
    def call(self, seq):
        local_features_input = self.conv(seq)
        # seq_input = tf.math.l2_normalize(self.prepend_task_token(local_features_input), axis=-1)
        seq_input = self.prepend_task_token(local_features_input)
        seq_output = self.transformer(seq_input)
        
        core_summary = seq_output[:, 0, :]
        core_seq = self.reconstructor(seq_output[:, 1:, :])
        
        if self.framewise:
            return core_seq
        else:
            return core_summary

    def get_visualization_token(self, seq):
        "Gets the first frame level token for projector visualization."
        local_features_input = self.conv(seq)
        # seq_input = tf.math.l2_normalize(self.prepend_task_token(local_features_input), axis=-1)
        seq_input = self.prepend_task_token(local_features_input)
        seq_output = self.transformer(seq_input)
        return seq_output[:, 1, :]


@keras.utils.register_keras_serializable('Attention')    
class Transformer(keras.layers.Layer):
    """Core transformer component for tokenized sequential data."""
    def __init__(self, depth, num_heads, inner_dim, use_layer_norm, dense_width_array, dense_activation, dropout):
        super().__init__()
        self.depth = depth
        self.num_heads = num_heads
        self.inner_dim = inner_dim
        self.use_layer_norm = use_layer_norm
        self.dense_width_array = dense_width_array
        self.dense_activation = dense_activation
        self.dropout = dropout

        self.position_embedding = PositionEmbedding()
        if self.use_layer_norm:
            self.layer_norm_layers = [keras.layers.LayerNormalization() for _ in range(2 * self.depth)]

        self.attention_layers = [keras.layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.inner_dim) for _ in range(self.depth)]
        self.attention_dropout = [keras.layers.Dropout(self.dropout) for _ in range(self.depth)]

        self.dense_layers = []
        self.dense_dropout = []
        for _ in range(self.depth):
            self.dense_layers.append([])
            self.dense_dropout.append([])
            for dense_unit in [*self.dense_width_array, self.inner_dim]:
                self.dense_layers[-1].append(keras.layers.Dense(units=dense_unit, activation=self.dense_activation))
                self.dense_dropout[-1].append(keras.layers.Dropout(self.dropout))

    def get_config(self):
        config = {
            "depth": self.depth,
            "num_heads": self.num_heads,
            "inner_dim": self.inner_dim,
            "use_layer_norm": self.use_layer_norm,
            "dense_width_array": self.dense_width_array,
            "dense_activation": self.dense_activation,
            "dropout": self.dropout,
        }
        return config
        
    def call(self, base_seq_input):
        seq_input = base_seq_input + self.position_embedding(base_seq_input)
        for depth in range(self.depth):
            seq_input_residule = seq_input
            
            if self.use_layer_norm:
                ln = self.layer_norm_layers[2 * depth]
                seq_input = ln(seq_input)

            attention = self.attention_layers[depth]
            dropout = self.attention_dropout[depth]
            seq_input = dropout(attention(seq_input, seq_input))

            seq_input += seq_input_residule
            seq_input_residule = seq_input

            if self.use_layer_norm:
                ln = self.layer_norm_layers[2 * depth + 1]
                seq_input = ln(seq_input)

            for dense_ff, dropout in zip(self.dense_layers[depth], self.dense_dropout[depth]):
                seq_input = dense_ff(seq_input)
                seq_input = dropout(seq_input)
                
            seq_input += seq_input_residule

            if not self.use_layer_norm:
                seq_input = tf.math.l2_normalize(seq_input, axis=-1)
        return seq_input


@keras.utils.register_keras_serializable('Attention') 
class PositionEmbedding(tf.keras.layers.Layer):
    """Creates a positional embedding."""

    def __init__(self,
               max_length=400,
               initializer="glorot_uniform",
               seq_axis=1,
               **kwargs):

        super().__init__(**kwargs)
        if max_length is None:
            raise ValueError(
              "`max_length` must be an Integer, not `None`."
            )
        self._max_length = max_length
        self._initializer = tf.keras.initializers.get(initializer)
        self._seq_axis = seq_axis

    def get_config(self):
        config = {
            "max_length": self._max_length,
            "initializer": tf.keras.initializers.serialize(self._initializer),
            "seq_axis": self._seq_axis,
        }
        return config

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