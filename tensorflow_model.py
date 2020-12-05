import tensorflow as tf
import numpy as np
import time
from typing import Dict, Optional, List, Iterable
from collections import Counter
from functools import partial

from ast_contexts import ASTContextReader, ModelInputFormer, ReaderInputTensors, EvaluatorAction
from common import common
from vocabularies import VocabType
from config import Config
from model_base import CodeVectorizeModelBase, ModelEvalResults, ModelForecastResults


tf.compat.v1.disable_eager_execution()


class CodeVectorizeModel(CodeVectorizeModelBase):
    def __init__(self, config: Config):
        self.sess = tf.compat.v1.Session()
        self.saver = None

        self.eval_reader = None
        self.eval_input_iterator_reset_op = None
        self.predict_reader = None

        self.predict_placeholder = None
        self.eval_top_words_op, self.eval_top_values_op, self.eval_original_names_op, self.eval_snippet_vectors = None, None, None, None
        self.predict_top_words_op, self.predict_top_values_op, self.predict_original_names_op = None, None, None

        self.vocab_type_to_tf_variable_name_mapping: Dict[VocabType, str] = {
            VocabType.Token: 'WORDS_VOCAB',
            VocabType.Target: 'TARGET_WORDS_VOCAB',
            VocabType.Path: 'PATHS_VOCAB'
        }

        super(CodeVectorizeModel, self).__init__(config)

    def train(self):
        self.log('Starting training')
        start_time = time.time()

        batch_num = 0
        sum_loss = 0
        multi_batch_start_time = time.time()
        num_batches_to_save_and_eval = max(int(self.config.train_steps_per_epoch * self.config.SAVE_EVERY_EPOCHS), 1)

        train_reader = ASTContextReader(vocabs=self.vocabs,
                                         model_input_tensors_former=_TFTrainModelInputFormer(),
                                         config=self.config, estimator_action=EvaluatorAction.Train)
        input_iterator = tf.compat.v1.data.make_initializable_iterator(train_reader.get_dataset())
        input_iterator_reset_op = input_iterator.initializer
        input_tensors = input_iterator.get_next()

        optimizer, train_loss = self._build_tf_training_graph(input_tensors)
        self.saver = tf.compat.v1.train.Saver(max_to_keep=self.config.MAX_TO_KEEP)

        self.log('Number of trainable params: {}'.format(
            np.sum([np.prod(v.get_shape().as_list()) for v in tf.compat.v1.trainable_variables()])))
        for variable in tf.compat.v1.trainable_variables():
            self.log("variable name: {} -- shape: {} -- #params: {}".format(
                variable.name, variable.get_shape(), np.prod(variable.get_shape().as_list())))

        self._initialize_session_variables()

        if self.config.MODEL_LOAD_PATH:
            self._load_inner_model(self.sess)

        self.sess.run(input_iterator_reset_op)
        time.sleep(1)
        self.log('Started reader...')
        try:
            while True:
                batch_num += 1

                _, batch_loss = self.sess.run([optimizer, train_loss])

                sum_loss += batch_loss
                if batch_num % self.config.NUM_BATCHES_TO_LOG_PROGRESS == 0:
                    self._trace_training(sum_loss, batch_num, multi_batch_start_time)
                    sum_loss = 0
                    multi_batch_start_time = time.time()
                if batch_num % num_batches_to_save_and_eval == 0:
                    epoch_num = int((batch_num / num_batches_to_save_and_eval) * self.config.SAVE_EVERY_EPOCHS)
                    model_save_path = self.config.MODEL_SAVE_PATH + '_iter' + str(epoch_num)
                    self.save(model_save_path)
                    self.log('Saved after %d epochs in: %s' % (epoch_num, model_save_path))
                    evaluation_results = self.evaluate()
                    evaluation_results_str = (str(evaluation_results).replace('topk', 'top{}'.format(
                        self.config.TOP_USED_CONSIDERED_DURING_PREDICTION)))
                    self.log('After {nr_epochs} epochs -- {evaluation_results}'.format(
                        nr_epochs=epoch_num,
                        evaluation_results=evaluation_results_str
                    ))
        except tf.errors.OutOfRangeError:
            pass

        self.log('Done training')

        if self.config.MODEL_SAVE_PATH:
            self._save_inner_model(self.config.MODEL_SAVE_PATH)
            self.log('Model saved in file: %s' % self.config.MODEL_SAVE_PATH)

        elapsed = int(time.time() - start_time)
        self.log("Training time: %sH:%sM:%sS\n" % ((elapsed // 60 // 60), (elapsed // 60) % 60, elapsed % 60))

    def evaluate(self) -> Optional[ModelEvalResults]:
        eval_start_time = time.time()
        if self.eval_reader is None:
            self.eval_reader = ASTContextReader(vocabs=self.vocabs,
                                                 model_input_tensors_former=_TFEvaluateModelInputFormer(),
                                                 config=self.config, estimator_action=EvaluatorAction.Evaluate)
            input_iterator = tf.compat.v1.data.make_initializable_iterator(self.eval_reader.get_dataset())
            self.eval_input_iterator_reset_op = input_iterator.initializer
            input_tensors = input_iterator.get_next()

            self.eval_top_words_op, self.eval_top_values_op, self.eval_original_names_op, _, _, _, _, \
                self.eval_snippet_vectors = self._build_tf_test_graph(input_tensors)
            if self.saver is None:
                self.saver = tf.compat.v1.train.Saver()

        if self.config.MODEL_LOAD_PATH and not self.config.TRAIN_DATA_TRACK_PREFIX:
            self._initialize_session_variables()
            self._load_inner_model(self.sess)
            if self.config.RELEASE:
                release_name = self.config.MODEL_LOAD_PATH + '.release'
                self.log('Releasing model, output model: %s' % release_name)
                self.saver.save(self.sess, release_name)
                return None

        with open('log.txt', 'w') as log_output_file:
            if self.config.EXPORT_SNIPPET_VECTORS:
                snippet_vectors_file = open(self.config.TEST_DATA_PATH + '.vectors', 'w')
            total_forecasts = 0
            total_forecast_batches = 0
            subtokens_evaluation_metric = SubtokensEvaluationMetric(
                partial(common.filter_impossible_names, self.vocabs.target_vocab.special_words))
            topk_accuracy_evaluation_metric = TopKAccuracyEvaluationMetric(
                self.config.TOP_USED_CONSIDERED_DURING_PREDICTION,
                partial(common.get_first_match, self.vocabs.target_vocab.special_words))
            start_time = time.time()

            self.sess.run(self.eval_input_iterator_reset_op)

            self.log('Starting evaluation')

            try:
                while True:
                    top_words, top_scores, original_names, snippet_vectors = self.sess.run(
                        [self.eval_top_words_op, self.eval_top_values_op,
                         self.eval_original_names_op, self.eval_snippet_vectors],
                    )

                    top_words = common.binary_to_string_matrix(top_words)
                    original_names = common.binary_to_string_list(original_names)

                    self._log_forecasts_during_evaluation(zip(original_names, top_words), log_output_file)
                    topk_accuracy_evaluation_metric.update_batch(zip(original_names, top_words))
                    subtokens_evaluation_metric.update_batch(zip(original_names, top_words))

                    total_forecasts += len(original_names)
                    total_forecast_batches += 1
                    if self.config.EXPORT_SNIPPET_VECTORS:
                        self._write_snippet_vectors(snippet_vectors_file, snippet_vectors)
                    if total_forecast_batches % self.config.NUM_BATCHES_TO_LOG_PROGRESS == 0:
                        elapsed = time.time() - start_time

                        self._trace_evaluation(total_forecasts, elapsed)
            except tf.errors.OutOfRangeError:
                pass
            self.log('Done evaluating, epoch reached')
            log_output_file.write(str(topk_accuracy_evaluation_metric.topk_correct_forecasts) + '\n')
        if self.config.EXPORT_SNIPPET_VECTORS:
            snippet_vectors_file.close()
        
        elapsed = int(time.time() - eval_start_time)
        self.log("Evaluation time: %sH:%sM:%sS" % ((elapsed // 60 // 60), (elapsed // 60) % 60, elapsed % 60))
        return ModelEvalResults(
            topk_acc=topk_accuracy_evaluation_metric.topk_correct_forecasts,
            subtoken_precision=subtokens_evaluation_metric.precision,
            subtoken_recall=subtokens_evaluation_metric.recall,
            subtoken_f1=subtokens_evaluation_metric.f1)

    def _build_tf_training_graph(self, input_tensors):
        input_tensors = _TFTrainModelInputFormer().from_model_input_form(input_tensors)

        with tf.compat.v1.variable_scope('model'):
            tokens_vocab = tf.compat.v1.get_variable(
                self.vocab_type_to_tf_variable_name_mapping[VocabType.Token],
                shape=(self.vocabs.token_vocab.size, self.config.TOKEN_EMBEDDINGS_SIZE), dtype=tf.float32,
                initializer=tf.compat.v1.initializers.variance_scaling(scale=1.0, mode='fan_out', distribution="uniform"))
            targets_vocab = tf.compat.v1.get_variable(
                self.vocab_type_to_tf_variable_name_mapping[VocabType.Target],
                shape=(self.vocabs.target_vocab.size, self.config.TARGET_EMBEDDINGS_SIZE), dtype=tf.float32,
                initializer=tf.compat.v1.initializers.variance_scaling(scale=1.0, mode='fan_out', distribution="uniform"))
            notice_param = tf.compat.v1.get_variable(
                'ATTENTION',
                shape=(self.config.BODY_VECTOR_SIZE, 1), dtype=tf.float32)
            paths_vocab = tf.compat.v1.get_variable(
                self.vocab_type_to_tf_variable_name_mapping[VocabType.Path],
                shape=(self.vocabs.track_vocab.size, self.config.TRACK_EMBEDDINGS_SIZE), dtype=tf.float32,
                initializer=tf.compat.v1.initializers.variance_scaling(scale=1.0, mode='fan_out', distribution="uniform"))

            snippet_vectors, _ = self._calculate_weighted_contexts(
                tokens_vocab, paths_vocab, notice_param, input_tensors.track_source_token_indices,
                input_tensors.track_indices, input_tensors.track_target_token_indices, input_tensors.context_valid_mask)

            logits = tf.matmul(snippet_vectors, targets_vocab, transpose_b=True)
            batch_size = tf.cast(tf.shape(input_tensors.target_index)[0], dtype=tf.float32)
            loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tf.reshape(input_tensors.target_index, [-1]),
                logits=logits)) / batch_size

            optimizer = tf.compat.v1.train.AdamOptimizer().minimize(loss)

        return optimizer, loss

    def _calculate_weighted_contexts(self, tokens_vocab, paths_vocab, notice_param, source_input, track_input,
                                     target_input, valid_mask, is_evaluating=False):
        source_word_embed = tf.nn.embedding_lookup(params=tokens_vocab, ids=source_input)
        track_embed = tf.nn.embedding_lookup(params=paths_vocab, ids=track_input)
        target_word_embed = tf.nn.embedding_lookup(params=tokens_vocab, ids=target_input)

        context_embed = tf.concat([source_word_embed, track_embed, target_word_embed],
                                  axis=-1)

        if not is_evaluating:
            context_embed = tf.nn.dropout(context_embed, rate=1-self.config.DROPOUT_KEEP_RATE)

        flat_embed = tf.reshape(context_embed, [-1, self.config.context_vector_size])
        transform_param = tf.compat.v1.get_variable(
            'TRANSFORM', shape=(self.config.context_vector_size, self.config.BODY_VECTOR_SIZE), dtype=tf.float32)

        flat_embed = tf.tanh(tf.matmul(flat_embed, transform_param))

        contexts_weights = tf.matmul(flat_embed, notice_param)
        batched_contexts_weights = tf.reshape(
            contexts_weights, [-1, self.config.MAX_CONTEXTS, 1])
        mask = tf.math.log(valid_mask)
        mask = tf.expand_dims(mask, axis=2)
        batched_contexts_weights += mask
        notice_weights = tf.nn.softmax(batched_contexts_weights, axis=1)

        batched_embed = tf.reshape(flat_embed, shape=[-1, self.config.MAX_CONTEXTS, self.config.BODY_VECTOR_SIZE])
        snippet_vectors = tf.reduce_sum(tf.multiply(batched_embed, notice_weights), axis=1)

        return snippet_vectors, notice_weights

    def _build_tf_test_graph(self, input_tensors, normalize_scores=False):
        with tf.compat.v1.variable_scope('model', reuse=self.get_should_reuse_variables()):
            tokens_vocab = tf.compat.v1.get_variable(
                self.vocab_type_to_tf_variable_name_mapping[VocabType.Token],
                shape=(self.vocabs.token_vocab.size, self.config.TOKEN_EMBEDDINGS_SIZE),
                dtype=tf.float32, trainable=False)
            targets_vocab = tf.compat.v1.get_variable(
                self.vocab_type_to_tf_variable_name_mapping[VocabType.Target],
                shape=(self.vocabs.target_vocab.size, self.config.TARGET_EMBEDDINGS_SIZE),
                dtype=tf.float32, trainable=False)
            notice_param = tf.compat.v1.get_variable(
                'ATTENTION', shape=(self.config.context_vector_size, 1),
                dtype=tf.float32, trainable=False)
            paths_vocab = tf.compat.v1.get_variable(
                self.vocab_type_to_tf_variable_name_mapping[VocabType.Path],
                shape=(self.vocabs.track_vocab.size, self.config.TRACK_EMBEDDINGS_SIZE),
                dtype=tf.float32, trainable=False)

            targets_vocab = tf.transpose(targets_vocab)

            input_tensors = _TFEvaluateModelInputFormer().from_model_input_form(input_tensors)

            snippet_vectors, notice_weights = self._calculate_weighted_contexts(
                tokens_vocab, paths_vocab, notice_param, input_tensors.track_source_token_indices,
                input_tensors.track_indices, input_tensors.track_target_token_indices,
                input_tensors.context_valid_mask, is_evaluating=True)

        scores = tf.matmul(snippet_vectors, targets_vocab)

        topk_candidates = tf.nn.top_k(scores, k=tf.minimum(
            self.config.TOP_USED_CONSIDERED_DURING_PREDICTION, self.vocabs.target_vocab.size))
        top_indices = topk_candidates.indices
        top_words = self.vocabs.target_vocab.lookup_word(top_indices)
        original_words = input_tensors.target_string
        top_scores = topk_candidates.values
        if normalize_scores:
            top_scores = tf.nn.softmax(top_scores)

        return top_words, top_scores, original_words, notice_weights, input_tensors.track_source_token_strings, \
               input_tensors.track_strings, input_tensors.track_target_token_strings, snippet_vectors

    def predict(self, predict_data_lines: Iterable[str]) -> List[ModelForecastResults]:
        if self.predict_reader is None:
            self.predict_reader = ASTContextReader(vocabs=self.vocabs,
                                                    model_input_tensors_former=_TFEvaluateModelInputFormer(),
                                                    config=self.config, estimator_action=EvaluatorAction.Predict)
            self.predict_placeholder = tf.compat.v1.placeholder(tf.string)
            reader_output = self.predict_reader.process_input_row(self.predict_placeholder)

            self.predict_top_words_op, self.predict_top_values_op, self.predict_original_names_op, \
            self.notice_weights_op, self.predict_source_string, self.predict_track_string, \
            self.predict_track_target_string, self.predict_snippet_vectors = \
                self._build_tf_test_graph(reader_output, normalize_scores=True)

            self._initialize_session_variables()
            self.saver = tf.compat.v1.train.Saver()
            self._load_inner_model(sess=self.sess)

        forecast_results: List[ModelForecastResults] = []
        for line in predict_data_lines:
            batch_top_words, batch_top_scores, batch_original_name, batch_notice_weights, batch_track_source_strings,\
                batch_track_strings, batch_track_target_strings, batch_snippet_vectors = self.sess.run(
                    [self.predict_top_words_op, self.predict_top_values_op, self.predict_original_names_op,
                     self.notice_weights_op, self.predict_source_string, self.predict_track_string,
                     self.predict_track_target_string, self.predict_snippet_vectors],
                    feed_dict={self.predict_placeholder: line})

            assert all(tensor.shape[0] == 1 for tensor in (batch_top_words, batch_top_scores, batch_original_name,
                                                           batch_notice_weights, batch_track_source_strings,
                                                           batch_track_strings, batch_track_target_strings,
                                                           batch_snippet_vectors))
            top_words = np.squeeze(batch_top_words, axis=0)
            top_scores = np.squeeze(batch_top_scores, axis=0)
            original_name = batch_original_name[0]
            notice_weights = np.squeeze(batch_notice_weights, axis=0)
            track_source_strings = np.squeeze(batch_track_source_strings, axis=0)
            track_strings = np.squeeze(batch_track_strings, axis=0)
            track_target_strings = np.squeeze(batch_track_target_strings, axis=0)
            snippet_vectors = np.squeeze(batch_snippet_vectors, axis=0)

            top_words = common.binary_to_string_list(top_words)
            original_name = common.binary_to_string(original_name)
            notice_per_context = self._get_notice_weight_per_context(
                track_source_strings, track_strings, track_target_strings, notice_weights)
            forecast_results.append(ModelForecastResults(
                original_name=original_name,
                used_predicted_words=top_words,
                used_predicted_words_scores=top_scores,
                notice_per_context=notice_per_context,
                code_vector=(snippet_vectors if self.config.EXPORT_SNIPPET_VECTORS else None)
            ))
        return forecast_results

    def _save_inner_model(self, path: str):
        self.saver.save(self.sess, path)

    def _load_inner_model(self, sess=None):
        if sess is not None:
            self.log('Loading model weights from: ' + self.config.MODEL_LOAD_PATH)
            self.saver.restore(sess, self.config.MODEL_LOAD_PATH)
            self.log('Done loading model weights')

    def _get_vocab_embedding(self, vocab_type: VocabType) -> np.ndarray:
        assert vocab_type in VocabType
        vocab_tf_variable_name = self.vocab_type_to_tf_variable_name_mapping[vocab_type]
        
        if self.eval_reader is None:
            self.eval_reader = ASTContextReader(vocabs=self.vocabs,
                                                 model_input_tensors_former=_TFEvaluateModelInputFormer(),
                                                 config=self.config, estimator_action=EvaluatorAction.Evaluate)
            input_iterator = tf.compat.v1.data.make_initializable_iterator(self.eval_reader.get_dataset())
            _, _, _, _, _, _, _, _ = self._build_tf_test_graph(input_iterator.get_next())

        if vocab_type is VocabType.Token:
            shape = (self.vocabs.token_vocab.size, self.config.TOKEN_EMBEDDINGS_SIZE)
        elif vocab_type is VocabType.Target:
            shape = (self.vocabs.target_vocab.size, self.config.TARGET_EMBEDDINGS_SIZE)
        elif vocab_type is VocabType.Path:
            shape = (self.vocabs.track_vocab.size, self.config.TRACK_EMBEDDINGS_SIZE)

        with tf.compat.v1.variable_scope('model', reuse=True):
            embeddings = tf.compat.v1.get_variable(vocab_tf_variable_name, shape=shape)
        self.saver = tf.compat.v1.train.Saver()
        self._initialize_session_variables() 
        self._load_inner_model(self.sess) 
        vocab_embedding_matrix = self.sess.run(embeddings)
        return vocab_embedding_matrix

    def get_should_reuse_variables(self):
        if self.config.TRAIN_DATA_TRACK_PREFIX:
            return True
        else:
            return None

    def _log_forecasts_during_evaluation(self, results, output_file):
        for original_name, top_predicted_words in results:
            found_match = common.get_first_match(
                self.vocabs.target_vocab.special_words, original_name, top_predicted_words)
            if found_match is not None:
                forecast_idx, predicted_word = found_match
                if forecast_idx == 0:
                    output_file.write('Original: ' + original_name + ', predicted 1st: ' + predicted_word + '\n')
                else:
                    output_file.write('\t\t predicted correctly at rank: ' + str(forecast_idx + 1) + '\n')
            else:
                output_file.write('No results for predicting: ' + original_name)

    def _trace_training(self, sum_loss, batch_num, multi_batch_start_time):
        multi_batch_elapsed = time.time() - multi_batch_start_time
        avg_loss = sum_loss / (self.config.NUM_BATCHES_TO_LOG_PROGRESS * self.config.TRAIN_BATCH_SIZE)
        throughput = self.config.TRAIN_BATCH_SIZE * self.config.NUM_BATCHES_TO_LOG_PROGRESS / \
                     (multi_batch_elapsed if multi_batch_elapsed > 0 else 1)
        self.log('Average loss at batch %d: %f, \tthroughput: %d samples/sec' % (
            batch_num, avg_loss, throughput))

    def _trace_evaluation(self, total_forecasts, elapsed):
        state_message = 'Evaluated %d examples...' % total_forecasts
        throughput_message = "Forecast throughput: %d samples/sec" % int(
            total_forecasts / (elapsed if elapsed > 0 else 1))
        self.log(state_message)
        self.log(throughput_message)

    def close_session(self):
        self.sess.close()

    def _initialize_session_variables(self):
        self.sess.run(tf.group(
            tf.compat.v1.global_variables_initializer(),
            tf.compat.v1.local_variables_initializer(),
            tf.compat.v1.tables_initializer()))
        self.log('Initalized variables')


class SubtokensEvaluationMetric:
    def __init__(self, filter_impossible_names_fn):
        self.nr_true_positives: int = 0
        self.nr_false_positives: int = 0
        self.nr_false_negatives: int = 0
        self.nr_forecasts: int = 0
        self.filter_impossible_names_fn = filter_impossible_names_fn

    def update_batch(self, results):
        for original_name, top_words in results:
            forecast = self.filter_impossible_names_fn(top_words)[0]
            original_subtokens = Counter(common.get_subtokens(original_name))
            predicted_subtokens = Counter(common.get_subtokens(forecast))
            self.nr_true_positives += sum(count for element, count in predicted_subtokens.items()
                                          if element in original_subtokens)
            self.nr_false_positives += sum(count for element, count in predicted_subtokens.items()
                                           if element not in original_subtokens)
            self.nr_false_negatives += sum(count for element, count in original_subtokens.items()
                                           if element not in predicted_subtokens)
            self.nr_forecasts += 1

    @property
    def true_positive(self):
        return self.nr_true_positives / self.nr_forecasts

    @property
    def false_positive(self):
        return self.nr_false_positives / self.nr_forecasts

    @property
    def false_negative(self):
        return self.nr_false_negatives / self.nr_forecasts

    @property
    def precision(self):
        return self.nr_true_positives / (self.nr_true_positives + self.nr_false_positives)

    @property
    def recall(self):
        return self.nr_true_positives / (self.nr_true_positives + self.nr_false_negatives)

    @property
    def f1(self):
        return 2 * self.precision * self.recall / (self.precision + self.recall)


class TopKAccuracyEvaluationMetric:
    def __init__(self, used: int, get_first_match_fn):
        self.used = used
        self.nr_correct_forecasts = np.zeros(self.used)
        self.nr_forecasts: int = 0
        self.get_first_match_fn = get_first_match_fn

    def update_batch(self, results):
        for original_name, top_predicted_words in results:
            self.nr_forecasts += 1
            found_match = self.get_first_match_fn(original_name, top_predicted_words)
            if found_match is not None:
                suggestion_idx, _ = found_match
                self.nr_correct_forecasts[suggestion_idx:self.used] += 1

    @property
    def topk_correct_forecasts(self):
        return self.nr_correct_forecasts / self.nr_forecasts


class _TFTrainModelInputFormer(ModelInputFormer):
    def to_model_input_form(self, input_tensors: ReaderInputTensors):
        return input_tensors.target_index, input_tensors.track_source_token_indices, input_tensors.track_indices, \
               input_tensors.track_target_token_indices, input_tensors.context_valid_mask

    def from_model_input_form(self, input_row) -> ReaderInputTensors:
        return ReaderInputTensors(
            target_index=input_row[0],
            track_source_token_indices=input_row[1],
            track_indices=input_row[2],
            track_target_token_indices=input_row[3],
            context_valid_mask=input_row[4]
        )


class _TFEvaluateModelInputFormer(ModelInputFormer):
    def to_model_input_form(self, input_tensors: ReaderInputTensors):
        return (input_tensors.target_string, input_tensors.track_source_token_indices, input_tensors.track_indices,
                input_tensors.track_target_token_indices, input_tensors.context_valid_mask,
                input_tensors.track_source_token_strings, input_tensors.track_strings,
                input_tensors.track_target_token_strings)

    def from_model_input_form(self, input_row) -> ReaderInputTensors:
        return ReaderInputTensors(
            target_string=input_row[0],
            track_source_token_indices=input_row[1],
            track_indices=input_row[2],
            track_target_token_indices=input_row[3],
            context_valid_mask=input_row[4],
            track_source_token_strings=input_row[5],
            track_strings=input_row[6],
            track_target_token_strings=input_row[7]
        )
