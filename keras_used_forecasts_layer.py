import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
from collections import namedtuple


TopUsedForecastsLayerResult = namedtuple('TopUsedForecastsLayerResult', ['words', 'scores'])


class TopUsedForecastsLayer(Layer):
    def __init__(self,
                 used: int,
                 index_to_word_table: tf.lookup.StaticHashTable,
                 **kwargs):
        kwargs['dtype'] = tf.string
        kwargs['trainable'] = False
        super(TopUsedForecastsLayer, self).__init__(**kwargs)
        self.used = used
        self.index_to_word_table = index_to_word_table

    def build(self, input_shape):
        if len(input_shape) < 2:
            raise ValueError("Input shape for TopUsedForecastsLayer should be of >= 2 dimensions.")
        if input_shape[-1] < self.used:
            raise ValueError("Last dimension of input shape for TopUsedForecastsLayer should be of >= `used`.")
        super(TopUsedForecastsLayer, self).build(input_shape)
        self.trainable = False

    def call(self, y_pred, **kwargs) -> TopUsedForecastsLayerResult:
        used_pred_scores, used_pred_indices = tf.nn.top_k(y_pred, k=self.used, sorted=True)
        used_pred_indices = tf.cast(used_pred_indices, dtype=self.index_to_word_table.key_dtype)
        used_pred_words = self.index_to_word_table.lookup(used_pred_indices)

        return TopUsedForecastsLayerResult(words=used_pred_words, scores=used_pred_scores)

    def compute_output_shape(self, input_shape):
        output_shape = tuple(input_shape[:-1]) + (self.used, )
        return output_shape, output_shape
