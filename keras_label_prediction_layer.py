import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.layers import Layer
import tensorflow.python.keras.backend as K
from typing import Optional, List, Callable
from functools import reduce
from common import common


class LabelForecastLayer(Layer):
    FilterType = Callable[[tf.Tensor, tf.Tensor], tf.Tensor]

    def __init__(self,
                 used: int,
                 index_to_word_table: tf.contrib.lookup.HashTable,
                 predicted_words_filters: Optional[List[FilterType]] = None,
                 **kwargs):
        kwargs['dtype'] = tf.string
        kwargs['trainable'] = False
        super(LabelForecastLayer, self).__init__(**kwargs)
        self.used = used
        self.index_to_word_table = index_to_word_table
        self.predicted_words_filters = predicted_words_filters

    def build(self, input_shape):
        if len(input_shape) != 2:
            raise ValueError("Input shape for LabelForecastLayer should be of 2 dimension.")
        super(LabelForecastLayer, self).build(input_shape)
        self.trainable = False

    def call(self, y_pred, **kwargs):
        y_pred.shape.assert_has_rank(2)
        used_pred_indices = tf.cast(tf.nn.top_k(y_pred, k=self.used).indices,
                                     dtype=self.index_to_word_table.key_dtype)
        predicted_target_words_strings = self.index_to_word_table.lookup(used_pred_indices)

        masks = []
        if self.predicted_words_filters is not None:
            masks = [fltr(used_pred_indices, predicted_target_words_strings) for fltr in self.predicted_words_filters]
        if masks:
            legal_predicted_mask = reduce(tf.logical_and, masks)
        else:
            legal_predicted_mask = tf.cast(tf.ones_like(used_pred_indices), dtype=tf.bool)

        first_legal_predicted_mask = common.tf_get_first_true(legal_predicted_mask)
        first_legal_predicted_idx = tf.where(first_legal_predicted_mask)
        first_legal_predicted_word_string = tf.gather_nd(predicted_target_words_strings,
                                                         first_legal_predicted_idx)

        forecast = tf.reshape(first_legal_predicted_word_string, [-1])
        return forecast

    def compute_output_shape(self, input_shape):
        return input_shape[0],  # (batch,)
