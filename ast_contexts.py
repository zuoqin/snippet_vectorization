import tensorflow as tf
from typing import Dict, Tuple, NamedTuple, Union, Optional, Iterable
from config import Config
from vocabularies import CodeVectorizeVocabs
import abc
from functools import reduce
from enum import Enum


class EvaluatorAction(Enum):
    Train = 'train'
    Evaluate = 'evaluate'
    Predict = 'predict'

    @property
    def is_train(self):
        return self is EvaluatorAction.Train

    @property
    def is_evaluate(self):
        return self is EvaluatorAction.Evaluate

    @property
    def is_predict(self):
        return self is EvaluatorAction.Predict

    @property
    def is_evaluate_or_predict(self):
        return self.is_evaluate or self.is_predict


class ReaderInputTensors(NamedTuple):
    """
    Used mostly for convenient-and-clear access to input parts (by their names).
    """
    track_source_token_indices: tf.Tensor
    track_indices: tf.Tensor
    track_target_token_indices: tf.Tensor
    context_valid_mask: tf.Tensor
    target_index: Optional[tf.Tensor] = None
    target_string: Optional[tf.Tensor] = None
    track_source_token_strings: Optional[tf.Tensor] = None
    track_strings: Optional[tf.Tensor] = None
    track_target_token_strings: Optional[tf.Tensor] = None


class ModelInputFormer(abc.ABC):
    @abc.abstractmethod
    def to_model_input_form(self, input_tensors: ReaderInputTensors):
        ...

    @abc.abstractmethod
    def from_model_input_form(self, input_row) -> ReaderInputTensors:
        ...


class ASTContextReader:
    def __init__(self,
                 vocabs: CodeVectorizeVocabs,
                 config: Config,
                 model_input_tensors_former: ModelInputFormer,
                 estimator_action: EvaluatorAction,
                 repeat_endlessly: bool = False):
        self.vocabs = vocabs
        self.config = config
        self.model_input_tensors_former = model_input_tensors_former
        self.estimator_action = estimator_action
        self.repeat_endlessly = repeat_endlessly
        self.CONTEXT_PADDING = ','.join([self.vocabs.token_vocab.special_words.PAD,
                                         self.vocabs.track_vocab.special_words.PAD,
                                         self.vocabs.token_vocab.special_words.PAD])
        self.csv_record_defaults = [[self.vocabs.target_vocab.special_words.OOV]] + \
                                   ([[self.CONTEXT_PADDING]] * self.config.MAX_CONTEXTS)

        self.create_needed_vocabs_lookup_tables(self.vocabs)

        self._dataset: Optional[tf.data.Dataset] = None

    @classmethod
    def create_needed_vocabs_lookup_tables(cls, vocabs: CodeVectorizeVocabs):
        vocabs.token_vocab.get_word_to_index_lookup_table()
        vocabs.track_vocab.get_word_to_index_lookup_table()
        vocabs.target_vocab.get_word_to_index_lookup_table()

    @tf.function
    def process_input_row(self, row_placeholder):
        parts = tf.io.decode_csv(
            row_placeholder, record_defaults=self.csv_record_defaults, field_delim=' ', use_quote_delim=False)

        tensors = self._map_raw_dataset_row_to_input_tensors(*parts)

        tensors_expanded = ReaderInputTensors(
            **{name: None if tensor is None else tf.expand_dims(tensor, axis=0)
               for name, tensor in tensors._asdict().items()})
        return self.model_input_tensors_former.to_model_input_form(tensors_expanded)

    def process_iterate_input(self, input_data_lines: Iterable) -> Iterable:
        for data_row in input_data_lines:
            processed_row = self.process_input_row(data_row)
            yield processed_row

    def get_dataset(self, input_data_rows: Optional = None) -> tf.data.Dataset:
        if self._dataset is None:
            self._dataset = self._create_dataset_pipeline(input_data_rows)
        return self._dataset

    def _create_dataset_pipeline(self, input_data_rows: Optional = None) -> tf.data.Dataset:
        if input_data_rows is None:
            assert not self.estimator_action.is_predict
            dataset = tf.data.experimental.CsvDataset(
                self.config.data_path(is_evaluating=self.estimator_action.is_evaluate),
                record_defaults=self.csv_record_defaults, field_delim=' ', use_quote_delim=False,
                buffer_size=self.config.CSV_BUFFER_SIZE)
        else:
            dataset = tf.data.Dataset.from_tensor_slices(input_data_rows)
            dataset = dataset.map(
                lambda input_line: tf.io.decode_csv(
                    tf.reshape(tf.cast(input_line, tf.string), ()),
                    record_defaults=self.csv_record_defaults,
                    field_delim=' ', use_quote_delim=False))

        if self.repeat_endlessly:
            dataset = dataset.repeat()
        if self.estimator_action.is_train:
            if not self.repeat_endlessly and self.config.NUM_TRAIN_EPOCHS > 1:
                dataset = dataset.repeat(self.config.NUM_TRAIN_EPOCHS)
            dataset = dataset.shuffle(self.config.SHUFFLE_BUFFER_SIZE, reshuffle_each_iteration=True)

        dataset = dataset.map(self._map_raw_dataset_row_to_expected_model_input_form,
                              num_parallel_calls=self.config.READER_NUM_PARALLEL_BATCHES)
        batch_size = self.config.batch_size(is_evaluating=self.estimator_action.is_evaluate)
        if self.estimator_action.is_predict:
            dataset = dataset.batch(1)
        else:
            dataset = dataset.filter(self._filter_input_rows)
            dataset = dataset.batch(batch_size)

        dataset = dataset.prefetch(buffer_size=40)
        return dataset

    def _filter_input_rows(self, *row_parts) -> tf.bool:
        row_parts = self.model_input_tensors_former.from_model_input_form(row_parts)

        any_word_valid_mask_per_context_part = [
            tf.not_equal(tf.reduce_max(row_parts.track_source_token_indices, axis=0),
                         self.vocabs.token_vocab.word_to_index[self.vocabs.token_vocab.special_words.PAD]),
            tf.not_equal(tf.reduce_max(row_parts.track_target_token_indices, axis=0),
                         self.vocabs.token_vocab.word_to_index[self.vocabs.token_vocab.special_words.PAD]),
            tf.not_equal(tf.reduce_max(row_parts.track_indices, axis=0),
                         self.vocabs.track_vocab.word_to_index[self.vocabs.track_vocab.special_words.PAD])]
        any_contexts_is_valid = reduce(tf.logical_or, any_word_valid_mask_per_context_part)  # scalar

        if self.estimator_action.is_evaluate:
            cond = any_contexts_is_valid
        else:
            word_is_valid = tf.greater(
                row_parts.target_index, self.vocabs.target_vocab.word_to_index[self.vocabs.target_vocab.special_words.OOV])  # scalar
            cond = tf.logical_and(word_is_valid, any_contexts_is_valid)  # scalar

        return cond

    def _map_raw_dataset_row_to_expected_model_input_form(self, *row_parts) -> \
            Tuple[Union[tf.Tensor, Tuple[tf.Tensor, ...], Dict[str, tf.Tensor]], ...]:
        tensors = self._map_raw_dataset_row_to_input_tensors(*row_parts)
        return self.model_input_tensors_former.to_model_input_form(tensors)

    def _map_raw_dataset_row_to_input_tensors(self, *row_parts) -> ReaderInputTensors:
        row_parts = list(row_parts)
        target_str = row_parts[0]
        target_index = self.vocabs.target_vocab.lookup_index(target_str)

        contexts_str = tf.stack(row_parts[1:(self.config.MAX_CONTEXTS + 1)], axis=0)
        split_contexts = tf.compat.v1.string_split(contexts_str, sep=',', skip_empty=False)

        sparse_split_contexts = tf.sparse.SparseTensor(
            indices=split_contexts.indices, values=split_contexts.values, dense_shape=[self.config.MAX_CONTEXTS, 3])
        dense_split_contexts = tf.reshape(
            tf.sparse.to_dense(sp_input=sparse_split_contexts, default_value=self.vocabs.token_vocab.special_words.PAD),
            shape=[self.config.MAX_CONTEXTS, 3])

        track_source_token_strings = tf.squeeze(
            tf.slice(dense_split_contexts, begin=[0, 0], size=[self.config.MAX_CONTEXTS, 1]), axis=1)
        track_strings = tf.squeeze(
            tf.slice(dense_split_contexts, begin=[0, 1], size=[self.config.MAX_CONTEXTS, 1]), axis=1)
        track_target_token_strings = tf.squeeze(
            tf.slice(dense_split_contexts, begin=[0, 2], size=[self.config.MAX_CONTEXTS, 1]), axis=1)

        track_source_token_indices = self.vocabs.token_vocab.lookup_index(track_source_token_strings)
        track_indices = self.vocabs.track_vocab.lookup_index(track_strings)  # (max_contexts, )
        track_target_token_indices = self.vocabs.token_vocab.lookup_index(track_target_token_strings)

        valid_word_mask_per_context_part = [
            tf.not_equal(track_source_token_indices, self.vocabs.token_vocab.word_to_index[self.vocabs.token_vocab.special_words.PAD]),
            tf.not_equal(track_target_token_indices, self.vocabs.token_vocab.word_to_index[self.vocabs.token_vocab.special_words.PAD]),
            tf.not_equal(track_indices, self.vocabs.track_vocab.word_to_index[self.vocabs.track_vocab.special_words.PAD])]  # [(max_contexts, )]
        context_valid_mask = tf.cast(reduce(tf.logical_or, valid_word_mask_per_context_part), dtype=tf.float32)  # (max_contexts, )

        return ReaderInputTensors(
            track_source_token_indices=track_source_token_indices,
            track_indices=track_indices,
            track_target_token_indices=track_target_token_indices,
            context_valid_mask=context_valid_mask,
            target_index=target_index,
            target_string=target_str,
            track_source_token_strings=track_source_token_strings,
            track_strings=track_strings,
            track_target_token_strings=track_target_token_strings
        )
