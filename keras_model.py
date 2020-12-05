import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Embedding, Concatenate, Dropout, TimeDistributed, Dense
from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as K
from tensorflow.keras.metrics import sparse_top_k_categorical_accuracy

from ast_contexts import ASTContextReader, ModelInputFormer, ReaderInputTensors, EvaluatorAction
import os
import numpy as np
from functools import partial
from typing import List, Optional, Iterable, Union, Callable, Dict
from collections import namedtuple
import time
import datetime
from vocabularies import VocabType
from keras_mindfull_layer import MindfullLayer
from keras_used_forecasts_layer import TopUsedForecastsLayer
from keras_partial_words_metrics import PartialWordsPrecisionMetric, PartialWordsRecallMetric, PartialWordsF1Metric
from config import Config
from common import common
from model_base import CodeVectorizeModelBase, ModelEvalResults, ModelForecastResults
from keras_checkpoint_callback import TrainingModelStatus, TrainingModelStatusTrackerCallback,\
    ModelCheckCallback, MultiBatchCallback, TrainingModelProgressLoggerCallback


class CodeVectorizeModel(CodeVectorizeModelBase):
    def __init__(self, config: Config):
        self.keras_train_model: Optional[keras.Model] = None
        self.keras_eval_model: Optional[keras.Model] = None
        self.keras_model_predict_function: Optional[K.GraphExecutionFunction] = None
        self.training_status: TrainingModelStatus = TrainingModelStatus()
        self._checkpoint: Optional[tf.train.Checkpoint] = None
        self._checkpoint_manager: Optional[tf.train.CheckpointManager] = None
        super(CodeVectorizeModel, self).__init__(config)

    def _create_keras_model(self):
        track_source_token = Input((self.config.MAX_CONTEXTS,), dtype=tf.int32)
        track_input = Input((self.config.MAX_CONTEXTS,), dtype=tf.int32)
        track_target_token = Input((self.config.MAX_CONTEXTS,), dtype=tf.int32)
        context_valid_mask = Input((self.config.MAX_CONTEXTS,))

        paths_embedded = Embedding(
            self.vocabs.track_vocab.size, self.config.TRACK_EMBEDDINGS_SIZE, name='path_embedding')(track_input)

        token_shared_layer = Embedding(
            self.vocabs.token_vocab.size, self.config.TOKEN_EMBEDDINGS_SIZE, name='token_embedding')
        track_source_token = token_shared_layer(track_source_token)
        track_target_token = token_shared_layer(track_target_token)

        context_embedded = Concatenate()([track_source_token, paths_embedded, track_target_token])
        context_embedded = Dropout(1 - self.config.DROPOUT_KEEP_RATE)(context_embedded)

        context_after_dense = TimeDistributed(
            Dense(self.config.BODY_VECTOR_SIZE, use_bias=False, activation='tanh'))(context_embedded)

        snippet_vectors, notice_weights = MindfullLayer(name='attention')(
            [context_after_dense, context_valid_mask])

        target_index = Dense(
            self.vocabs.target_vocab.size, use_bias=False, activation='softmax', name='target_index')(snippet_vectors)

        inputs = [track_source_token, track_input, track_target_token, context_valid_mask]
        self.keras_train_model = keras.Model(inputs=inputs, outputs=target_index)

        used_predicted_words, used_predicted_words_scores = TopUsedForecastsLayer(
            self.config.TOP_USED_CONSIDERED_DURING_PREDICTION,
            self.vocabs.target_vocab.get_index_to_word_lookup_table(),
            name='target_string')(target_index)

        self.keras_eval_model = keras.Model(
            inputs=inputs, outputs=[target_index, used_predicted_words], name="code-vectorize-keras-model")

        predict_outputs = tuple(KerasForecastModelOutput(
            target_index=target_index, snippet_vectors=snippet_vectors, notice_weights=notice_weights,
            used_predicted_words=used_predicted_words, used_predicted_words_scores=used_predicted_words_scores))
        self.keras_model_predict_function = K.function(inputs=inputs, outputs=predict_outputs)

    def _create_metrics_for_keras_eval(self) -> Dict[str, List[Union[Callable, keras.metrics.Metric]]]:
        used_acc_metrics = []
        for k in range(1, self.config.TOP_USED_CONSIDERED_DURING_PREDICTION + 1):
            used_acc_metric = partial(
                sparse_top_k_categorical_accuracy, k=k)
            used_acc_metric.__name__ = 'top{k}_acc'.format(k=k)
            used_acc_metrics.append(used_acc_metric)
        predicted_words_filters = [
            lambda word_strings: tf.not_equal(word_strings, self.vocabs.target_vocab.special_words.OOV),
            lambda word_strings: tf.strings.regex_full_match(word_strings, r'^[a-zA-Z\|]+$')
        ]
        words_subtokens_metrics = [
            PartialWordsPrecisionMetric(predicted_words_filters=predicted_words_filters, name='subtoken_precision'),
            PartialWordsRecallMetric(predicted_words_filters=predicted_words_filters, name='subtoken_recall'),
            PartialWordsF1Metric(predicted_words_filters=predicted_words_filters, name='subtoken_f1')
        ]
        return {'target_index': used_acc_metrics, 'target_string': words_subtokens_metrics}

    @classmethod
    def _create_optimizer(cls):
        return tf.optimizers.Adam()

    def _compile_keras_model(self, optimizer=None):
        if optimizer is None:
            optimizer = self.keras_train_model.optimizer
            if optimizer is None:
                optimizer = self._create_optimizer()

        def zero_loss(true_word, topk_forecasts):
            return tf.constant(0.0, shape=(), dtype=tf.float32)

        self.keras_train_model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=optimizer)

        self.keras_eval_model.compile(
            loss={'target_index': 'sparse_categorical_crossentropy', 'target_string': zero_loss},
            optimizer=optimizer,
            metrics=self._create_metrics_for_keras_eval())

    def _create_data_reader(self, estimator_action: EvaluatorAction, repeat_endlessly: bool = False):
        return ASTContextReader(
            vocabs=self.vocabs,
            config=self.config,
            model_input_tensors_former=_KerasModelInputFormer(estimator_action=estimator_action),
            estimator_action=estimator_action,
            repeat_endlessly=repeat_endlessly)

    def _create_train_callbacks(self) -> List[Callback]:
        keras_callbacks = [
            TrainingModelStatusTrackerCallback(self.training_status),
            TrainingModelProgressLoggerCallback(self.config, self.training_status),
        ]
        if self.config.is_saving:
            keras_callbacks.append(ModelCheckCallback(
                self, self.config.SAVE_EVERY_EPOCHS, self.logger))
        if self.config.is_testing:
            keras_callbacks.append(ModelEvalCallback(self))
        if self.config.USE_TENSORBOARD:
            log_dir = "logs/scalars/train_" + common.now_str()
            tensorboard_callback = keras.callbacks.TensorBoard(
                log_dir=log_dir,
                update_freq=self.config.NUM_BATCHES_TO_LOG_PROGRESS * self.config.TRAIN_BATCH_SIZE)
            keras_callbacks.append(tensorboard_callback)
        return keras_callbacks

    def train(self):
        train_data_input_reader = self._create_data_reader(estimator_action=EvaluatorAction.Train)

        training_history = self.keras_train_model.fit(
            train_data_input_reader.get_dataset(),
            steps_per_epoch=self.config.train_steps_per_epoch,
            epochs=self.config.NUM_TRAIN_EPOCHS,
            initial_epoch=self.training_status.nr_epochs_trained,
            verbose=self.config.VERBOSE_MODE,
            callbacks=self._create_train_callbacks())

        self.log(training_history)

    def evaluate(self) -> Optional[ModelEvalResults]:
        val_data_input_reader = self._create_data_reader(estimator_action=EvaluatorAction.Evaluate)
        eval_res = self.keras_eval_model.evaluate(
            val_data_input_reader.get_dataset(),
            steps=self.config.test_steps,
            verbose=self.config.VERBOSE_MODE)
        k = self.config.TOP_USED_CONSIDERED_DURING_PREDICTION
        return ModelEvalResults(
            topk_acc=eval_res[3:k+3],
            subtoken_precision=eval_res[k+3],
            subtoken_recall=eval_res[k+4],
            subtoken_f1=eval_res[k+5],
            loss=eval_res[1]
        )

    def predict(self, predict_data_rows: Iterable[str]) -> List[ModelForecastResults]:
        predict_input_read = self._create_data_reader(estimator_action=EvaluatorAction.Predict)
        input_iterator = predict_input_read.process_iterate_input(predict_data_rows)
        all_model_forecast_results = []
        for input_row in input_iterator:
            input_for_predict = input_row[0][:4]
            forecast_results = self.keras_model_predict_function(input_for_predict)

            forecast_results = KerasForecastModelOutput(
                *common.squeeze_single_batch(forecast_results))
            input_row = _KerasModelInputFormer(
                estimator_action=EvaluatorAction.Predict).from_model_input_form(input_row)
            input_row = ReaderInputTensors(*common.squeeze_single_batch(input_row))

            # calculate the notice weight for each context
            notice_per_context = self._get_notice_weight_per_context(
                track_source_strings=input_row.track_source_token_strings,
                track_strings=input_row.track_strings,
                track_target_strings=input_row.track_target_token_strings,
                notice_weights=forecast_results.notice_weights
            )

            # store the calculated forecast results in the wanted format.
            model_forecast_results = ModelForecastResults(
                original_name=common.binary_to_string(input_row.target_string.item()),
                used_predicted_words=common.binary_to_string_list(forecast_results.used_predicted_words),
                used_predicted_words_scores=forecast_results.used_predicted_words_scores,
                notice_per_context=notice_per_context,
                code_vector=forecast_results.snippet_vectors)
            all_model_forecast_results.append(model_forecast_results)

        return all_model_forecast_results

    def _save_inner_model(self, path):
        if self.config.RELEASE:
            self.keras_train_model.save_weights(self.config.get_model_weights_path(path))
        else:
            self._get_checkpoint_manager().save(checkpoint_number=self.training_status.nr_epochs_trained)

    def _create_inner_model(self):
        self._create_keras_model()
        self._compile_keras_model()
        self.keras_train_model.summary(print_fn=self.log)

    def _load_inner_model(self):
        self._create_keras_model()
        self._compile_keras_model()

        must_use_entire_model = self.config.is_training
        entire_model_exists = os.path.exists(self.config.entire_model_load_path)
        model_weights_exist = os.path.exists(self.config.model_weights_load_path)
        use_full_model = must_use_entire_model or not model_weights_exist

        if must_use_entire_model and not entire_model_exists:
            raise ValueError(
                "There is no model at path `{model_file_path}`. When loading the model for further training, "
                "we must use an entire saved model file (not just weights).".format(
                    model_file_path=self.config.entire_model_load_path))
        if not entire_model_exists and not model_weights_exist:
            raise ValueError(
                "There is no entire model to load at path `{entire_model_path}`, "
                "and there is no model weights file to load at path `{model_weights_path}`.".format(
                    entire_model_path=self.config.entire_model_load_path,
                    model_weights_path=self.config.model_weights_load_path))

        if use_full_model:
            self.log('Loading entire model from path `{}`.'.format(self.config.entire_model_load_path))
            latest_checkpoint = tf.train.latest_checkpoint(self.config.entire_model_load_path)
            if latest_checkpoint is None:
                raise ValueError("Failed to load model: Model latest checkpoint is not found.")
            self.log('Loading latest checkpoint `{}`.'.format(latest_checkpoint))
            status = self._get_checkpoint().restore(latest_checkpoint)
            status.initialize_or_restore()
            self.training_status.nr_epochs_trained = int(latest_checkpoint.split('-')[-1])
        else:
            self.log('Loading model weights from path `{}`.'.format(self.config.model_weights_load_path))
            self.keras_train_model.load_weights(self.config.model_weights_load_path)

        self.keras_train_model.summary(print_fn=self.log)

    def _get_checkpoint(self):
        assert self.keras_train_model is not None and self.keras_train_model.optimizer is not None
        if self._checkpoint is None:
            self._checkpoint = tf.train.Checkpoint(
                optimizer=self.keras_train_model.optimizer, model=self.keras_train_model)
        return self._checkpoint

    def _get_checkpoint_manager(self):
        if self._checkpoint_manager is None:
            self._checkpoint_manager = tf.train.CheckpointManager(
                self._get_checkpoint(), self.config.entire_model_save_path,
                max_to_keep=self.config.MAX_TO_KEEP)
        return self._checkpoint_manager

    def _get_vocab_embedding(self, vocab_type: VocabType) -> np.ndarray:
        assert vocab_type in VocabType

        vocab_type_to_embedding_layer_mapping = {
            VocabType.Target: 'target_index',
            VocabType.Token: 'token_embedding',
            VocabType.Path: 'path_embedding'
        }
        embedding_layer_name = vocab_type_to_embedding_layer_mapping[vocab_type]
        weight = np.array(self.keras_train_model.get_layer(embedding_layer_name).get_weights()[0])
        assert len(weight.shape) == 2

        assert self.vocabs.get(vocab_type).size in weight.shape
        if self.vocabs.get(vocab_type).size != weight.shape[0]:
            weight = np.transpose(weight)

        return weight

    def _create_lookup_tables(self):
        ASTContextReader.create_needed_vocabs_lookup_tables(self.vocabs)
        self.log('Lookup tables created.')

    def _initialize(self):
        self._create_lookup_tables()


class ModelEvalCallback(MultiBatchCallback):
    def __init__(self, code_vectorize_model: 'CodeVectorizeModel'):
        self.code_vectorize_model = code_vectorize_model
        self.avg_eval_duration: Optional[int] = None
        super(ModelEvalCallback, self).__init__(self.code_vectorize_model.config.NUM_TRAIN_BATCHES_TO_EVALUATE)

    def on_epoch_end(self, epoch, logs=None):
        self.perform_evaluation()

    def on_multi_batch_end(self, batch, logs, multi_batch_elapsed):
        self.perform_evaluation()

    def perform_evaluation(self):
        if self.avg_eval_duration is None:
            self.code_vectorize_model.log('Evaluating...')
        else:
            self.code_vectorize_model.log('Evaluating... (takes ~{})'.format(
                str(datetime.timedelta(seconds=int(self.avg_eval_duration)))))
        eval_start_time = time.time()
        evaluation_results = self.code_vectorize_model.evaluate()
        eval_duration = time.time() - eval_start_time
        if self.avg_eval_duration is None:
            self.avg_eval_duration = eval_duration
        else:
            self.avg_eval_duration = eval_duration * 0.5 + self.avg_eval_duration * 0.5
        self.code_vectorize_model.log('Done evaluating (took {}). Evaluation results:'.format(
            str(datetime.timedelta(seconds=int(eval_duration)))))

        self.code_vectorize_model.log(
            '    loss: {loss:.4f}, f1: {f1:.4f}, recall: {recall:.4f}, precision: {precision:.4f}'.format(
                loss=evaluation_results.loss, f1=evaluation_results.subtoken_f1,
                recall=evaluation_results.subtoken_recall, precision=evaluation_results.subtoken_precision))
        used_acc_formated = ['top{}: {:.4f}'.format(i, acc) for i, acc in enumerate(evaluation_results.topk_acc, start=1)]
        for used_acc_chunk in common.chunks(used_acc_formated, 5):
            self.code_vectorize_model.log('    ' + (', '.join(used_acc_chunk)))


class _KerasModelInputFormer(ModelInputFormer):
    def __init__(self, estimator_action: EvaluatorAction):
        self.estimator_action = estimator_action

    def to_model_input_form(self, input_tensors: ReaderInputTensors):
        inputs = (input_tensors.track_source_token_indices, input_tensors.track_indices,
                  input_tensors.track_target_token_indices, input_tensors.context_valid_mask)
        if self.estimator_action.is_train:
            targets = input_tensors.target_index
        else:
            targets = {'target_index': input_tensors.target_index, 'target_string': input_tensors.target_string}
        if self.estimator_action.is_predict:
            inputs += (input_tensors.track_source_token_strings, input_tensors.track_strings,
                       input_tensors.track_target_token_strings)
        return inputs, targets

    def from_model_input_form(self, input_row) -> ReaderInputTensors:
        inputs, targets = input_row
        return ReaderInputTensors(
            track_source_token_indices=inputs[0],
            track_indices=inputs[1],
            track_target_token_indices=inputs[2],
            context_valid_mask=inputs[3],
            target_index=targets if self.estimator_action.is_train else targets['target_index'],
            target_string=targets['target_string'] if not self.estimator_action.is_train else None,
            track_source_token_strings=inputs[4] if self.estimator_action.is_predict else None,
            track_strings=inputs[5] if self.estimator_action.is_predict else None,
            track_target_token_strings=inputs[6] if self.estimator_action.is_predict else None
        )


KerasForecastModelOutput = namedtuple(
    'KerasModelOutput', ['target_index', 'snippet_vectors', 'notice_weights',
                         'used_predicted_words', 'used_predicted_words_scores'])
