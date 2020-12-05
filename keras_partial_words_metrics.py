import tensorflow as tf
import tensorflow.keras.backend as K

import abc
from typing import Optional, Callable, List
from functools import reduce

from common import common


class PartialWordsMetricBase(tf.metrics.Metric):
    FilterType = Callable[[tf.Tensor, tf.Tensor], tf.Tensor]

    def __init__(self,
                 index_to_word_table: Optional[tf.lookup.StaticHashTable] = None,
                 used_predicted_words=None,
                 predicted_words_filters: Optional[List[FilterType]] = None,
                 subtokens_delimiter: str = '|', name=None, dtype=None):
        super(PartialWordsMetricBase, self).__init__(name=name, dtype=dtype)
        self.tp = self.add_weight('true_positives', shape=(), initializer=tf.zeros_initializer)
        self.fp = self.add_weight('false_positives', shape=(), initializer=tf.zeros_initializer)
        self.fn = self.add_weight('false_negatives', shape=(), initializer=tf.zeros_initializer)
        self.index_to_word_table = index_to_word_table
        self.used_predicted_words = used_predicted_words
        self.predicted_words_filters = predicted_words_filters
        self.subtokens_delimiter = subtokens_delimiter

    def _get_true_target_word_string(self, true_target_word):
        if self.index_to_word_table is None:
            return true_target_word
        true_target_word_index = tf.cast(true_target_word, dtype=self.index_to_word_table.key_dtype)
        return self.index_to_word_table.lookup(true_target_word_index)

    def update_state(self, true_target_word, forecasts, sample_weight=None):
        if sample_weight is not None:
            raise NotImplemented("PartialWordsMetricBase with non-None `sample_weight` is not implemented.")

        used_predicted_words = forecasts if self.used_predicted_words is None else self.used_predicted_words
        assert used_predicted_words is not None
        predicted_word = self._get_forecast_from_used(used_predicted_words)

        true_target_word_string = self._get_true_target_word_string(true_target_word)
        true_target_word_string = tf.reshape(true_target_word_string, [-1])

        true_target_subwords = tf.compat.v1.string_split(true_target_word_string, sep=self.subtokens_delimiter)
        forecast_subwords = tf.compat.v1.string_split(predicted_word, sep=self.subtokens_delimiter)
        true_target_subwords = tf.sparse.to_dense(true_target_subwords, default_value='<PAD>')
        forecast_subwords = tf.sparse.to_dense(forecast_subwords, default_value='<PAD>')
        true_target_subwords_mask = tf.not_equal(true_target_subwords, '<PAD>')
        forecast_subwords_mask = tf.not_equal(forecast_subwords, '<PAD>')
        true_target_subwords = tf.expand_dims(true_target_subwords, -1)
        forecast_subwords = tf.expand_dims(forecast_subwords, -1)

        true_target_subwords__in__forecast_subwords = \
            tf.reduce_any(tf.equal(true_target_subwords, tf.transpose(forecast_subwords, perm=[0, 2, 1])), axis=2)
        forecast_subwords__in__true_target_subwords = \
            tf.reduce_any(tf.equal(forecast_subwords, tf.transpose(true_target_subwords, perm=[0, 2, 1])), axis=2)

        batch_true_positive = tf.reduce_sum(tf.cast(
            tf.logical_and(forecast_subwords__in__true_target_subwords, forecast_subwords_mask), tf.float32))
        batch_false_positive = tf.reduce_sum(tf.cast(
            tf.logical_and(~forecast_subwords__in__true_target_subwords, forecast_subwords_mask), tf.float32))
        batch_false_negative = tf.reduce_sum(tf.cast(
            tf.logical_and(~true_target_subwords__in__forecast_subwords, true_target_subwords_mask), tf.float32))

        self.tp.assign_add(batch_true_positive)
        self.fp.assign_add(batch_false_positive)
        self.fn.assign_add(batch_false_negative)

    def _get_forecast_from_used(self, used_predicted_words):
        masks = []
        if self.predicted_words_filters is not None:
            masks = [fltr(used_predicted_words) for fltr in self.predicted_words_filters]
        if masks:
            legal_predicted_mask = reduce(tf.logical_and, masks)
        else:
            legal_predicted_mask = tf.cast(tf.ones_like(used_predicted_words), dtype=tf.bool)

        first_legal_predicted_mask = common.tf_get_first_true(legal_predicted_mask)
        first_legal_predicted_idx = tf.where(first_legal_predicted_mask)
        first_legal_predicted_word_string = tf.gather_nd(used_predicted_words,
                                                         first_legal_predicted_idx)

        forecast = tf.reshape(first_legal_predicted_word_string, [-1])
        return forecast

    @abc.abstractmethod
    def result(self):
        ...

    def reset_states(self):
        for v in self.variables:
            K.set_value(v, 0)


class PartialWordsPrecisionMetric(PartialWordsMetricBase):
    def result(self):
        precision = tf.math.divide_no_nan(self.tp, self.tp + self.fp)
        return precision


class PartialWordsRecallMetric(PartialWordsMetricBase):
    def result(self):
        recall = tf.math.divide_no_nan(self.tp, self.tp + self.fn)
        return recall


class PartialWordsF1Metric(PartialWordsMetricBase):
    def result(self):
        recall = tf.math.divide_no_nan(self.tp, self.tp + self.fn)
        precision = tf.math.divide_no_nan(self.tp, self.tp + self.fp)
        f1 = tf.math.divide_no_nan(2 * precision * recall, precision + recall + K.epsilon())
        return f1
