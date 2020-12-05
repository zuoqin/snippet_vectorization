import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras import backend as K
from typing import Optional


class MindfullLayer(Layer):
    def __init__(self, **kwargs):
        super(MindfullLayer, self).__init__(**kwargs)

    def build(self, inputs_shape):
        inputs_shape = inputs_shape if isinstance(inputs_shape, list) else [inputs_shape]

        if len(inputs_shape) < 1 or len(inputs_shape) > 2:
            raise ValueError("MindfullLayer expect one or two inputs.")

        input_shape = inputs_shape[0]

        if len(input_shape) != 3:
            raise ValueError("Input shape for MindfullLayer should be of 3 dimension.")

        self.input_length = int(input_shape[1])
        self.input_dim = int(input_shape[2])
        notice_param_shape = (self.input_dim, 1)

        self.notice_param = self.add_weight(
            name='notice_param',
            shape=notice_param_shape,
            initializer='uniform',
            trainable=True,
            dtype=tf.float32)
        super(MindfullLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        inputs = inputs if isinstance(inputs, list) else [inputs]

        if len(inputs) < 1 or len(inputs) > 2:
            raise ValueError("MindfullLayer expect one or two inputs.")

        actual_input = inputs[0]
        mask = inputs[1] if len(inputs) > 1 else None
        if mask is not None and not (((len(mask.shape) == 3 and mask.shape[2] == 1) or len(mask.shape) == 2)
                                     and mask.shape[1] == self.input_length):
            raise ValueError("`mask` should be of shape (batch, input_length) or (batch, input_length, 1) "
                             "when calling an MindfullLayer.")

        assert actual_input.shape[-1] == self.notice_param.shape[0]

        notice_weights = K.dot(actual_input, self.notice_param)

        if mask is not None:
            if len(mask.shape) == 2:
                mask = K.expand_dims(mask, axis=2)
            mask = K.log(mask)
            notice_weights += mask

        notice_weights = K.softmax(notice_weights, axis=1)
        result = K.sum(actual_input * notice_weights, axis=1)
        return result, notice_weights

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]  # (batch, input_dim)
