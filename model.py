#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
A model for named entity recognition.
"""
import logging
from datetime import datetime

import tensorflow as tf

from data_util import get_chunks
from defs import LBLS
from util import ConfusionMatrix, Progbar, minibatches

logger = logging.getLogger("NER")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

class Config:
    """
    Holds model hyperparams and data information.
    """
    n_word_features = 2
    window_size = 1
    # Number of features for every word in the input
    n_features = (2 * window_size + 1) * n_word_features
    # longest sequence to parse
    max_length = 120
    n_classes = 5
    dropout = 0.25
    embed_size = 50
    hidden_size = 100
    batch_size = 20
    n_epochs = 20
    max_grad_norm = 10.
    lr = 0.001

    def __init__(self, args):
        self.cell = args.cell

        if "model_path" in args:
            # Where to save things.
            self.output_path = args.model_path
        else:
            self.output_path = "results/{}/{:%Y%m%d_%H%M%S}/".format(
                self.cell, datetime.now())
        self.model_output = self.output_path + "model.weights"
        self.eval_output = self.output_path + "results.txt"
        self.conll_output = self.output_path + \
            "{}_predictions.conll".format(self.cell)
        self.log_output = self.output_path + "log"


class Model(object):
    """Abstracts a Tensorflow graph for a learning task.

    We use various Model classes as usual abstractions to encapsulate tensorflow
    computational graphs. Each algorithm you will construct in this homework will
    inherit from a Model object.
    """

    def add_placeholders(self):
        """
        Adds placeholder variables to tensorflow computational graph.
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def create_feed_dict(self, inputs_batch, labels_batch=None):
        """
        Creates the feed_dict for one step of training.
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def add_prediction_op(self):
        """
        Implements the core of the model that transforms a batch of input data into predictions.
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def add_loss_op(self, pred):
        """
        Adds Ops for the loss function to the computational graph.
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def add_training_op(self, loss):
        """
        Sets up the training Ops.
        """

        raise NotImplementedError("Each Model must re-implement this method.")

    def train_on_batch(self, sess, inputs_batch, labels_batch):
        """
        Perform one step of gradient descent on the provided batch of data.
        """
        feed = self.create_feed_dict(inputs_batch, labels_batch=labels_batch)
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss

    def predict_on_batch(self, sess, inputs_batch):
        """
        Make predictions for the provided batch of data
        """
        feed = self.create_feed_dict(inputs_batch)
        predictions = sess.run(self.pred, feed_dict=feed)
        return predictions

    def build(self):
        self.add_placeholders()
        self.pred = self.add_prediction_op()
        self.loss = self.add_loss_op(self.pred)
        self.train_op = self.add_training_op(self.loss)


class NERModel(Model):
    """
    Implements special functionality for NER models.
    """

    def __init__(self, helper, config, pretrained_embeddings, report=None):
        self.helper = helper
        self.report = report
        self.config = config
        self.max_length = min(Config.max_length, helper.max_length)
        Config.max_length = self.max_length
        self.pretrained_embeddings = pretrained_embeddings

        # Defining placeholders.
        self.input_placeholder = None
        self.labels_placeholder = None
        self.mask_placeholder = None
        self.dropout_placeholder = None

        self.build()

    def add_placeholders(self):
        """
        Generates placeholder variables to represent the input tensors
        """
        self.input_placeholder = tf.placeholder(dtype=tf.int32, shape=(
            None, self.max_length, self.config.n_features))
        self.labels_placeholder = tf.placeholder(
            dtype=tf.int32, shape=(None, self.max_length))
        self.mask_placeholder = tf.placeholder(
            dtype=tf.bool, shape=(None, self.max_length))
        self.dropout_placeholder = tf.placeholder(dtype=tf.float32)

    def create_feed_dict(self, inputs_batch, mask_batch, labels_batch=None, dropout=1):
        """
        Creates the feed_dict for the dependency parser.
        """
        feed_dict = {
            self.input_placeholder: inputs_batch,
            self.mask_placeholder: mask_batch,
            self.dropout_placeholder: dropout
        }
        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch
        return feed_dict

    def add_embedding(self):
        """
        Adds an embedding layer that maps from input tokens (integers) to vectors and then
        """
        word_embeddings = tf.Variable(self.pretrained_embeddings)
        embeddings = tf.nn.embedding_lookup(
            word_embeddings, self.input_placeholder)
        embeddings = tf.reshape(embeddings, shape=(
            -1, self.max_length, self.config.n_features*self.config.embed_size))
        return embeddings

    def add_prediction_op(self):
        """
        Adds bi-directional LSTM
        """
        # Get the word emedding of input
        x = self.add_embedding()

        # Create bi-directional LSTM
        forward_cell = tf.contrib.rnn.BasicLSTMCell(self.config.hidden_size)
        forward_cell = tf.contrib.rnn.DropoutWrapper(forward_cell,
                                                  input_keep_prob=1-self.config.dropout,
                                                  output_keep_prob=1-self.config.dropout)
        backward_cell = tf.contrib.rnn.BasicLSTMCell(self.config.hidden_size)
        backward_cell = tf.contrib.rnn.DropoutWrapper(backward_cell,
                                                  input_keep_prob=1-self.config.dropout,
                                                  output_keep_prob=1-self.config.dropout)
        outputs, state = tf.nn.bidirectional_dynamic_rnn(forward_cell, backward_cell, x, dtype=tf.float32)
        
        # Feed the output to a fully connected layer
        outputs = tf.concat(outputs, axis=2)
        preds = tf.contrib.layers.fully_connected(
            outputs, self.config.n_classes, activation_fn=None)

        return preds

    def add_loss_op(self, preds):
        """
        Adds Ops for the loss function to the computational graph.
        """
        # seq_length = tf.convert_to_tensor(
        #     self.config.batch_size * [self.max_length], dtype=tf.int32)
        # log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(
        #     inputs=preds, tag_indices=self.labels_placeholder,
        #     sequence_lengths=seq_length
        # )
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.boolean_mask(
                self.labels_placeholder, self.mask_placeholder),
            logits=tf.boolean_mask(preds, self.mask_placeholder))
        loss = tf.reduce_mean(loss)
        return loss

    def add_training_op(self, loss):
        """
        Sets up the training Ops.
        """
        optimizer = tf.train.AdamOptimizer(learning_rate=self.config.lr)
        train_op = optimizer.minimize(loss)
        return train_op

    def preprocess_sequence_data(self, examples):
        def featurize_windows(data, start, end, window_size=1):
            """Uses the input sequences in @data to construct new windowed data points.
            """
            ret = []
            for sentence, labels in data:
                from util import window_iterator
                sentence_ = []
                for window in window_iterator(sentence, window_size, beg=start, end=end):
                    sentence_.append(sum(window, []))
                ret.append((sentence_, labels))
            return ret

        examples = featurize_windows(
            examples, self.helper.START, self.helper.END)
        return pad_sequences(examples, self.max_length)

    def consolidate_predictions(self, examples_raw, examples, preds):
        """
        Batch the predictions into groups of sentence length.
        """
        assert len(examples_raw) == len(examples)
        assert len(examples_raw) == len(preds)

        ret = []
        for i, (sentence, labels) in enumerate(examples_raw):
            _, _, mask = examples[i]
            # only select elements of mask.
            labels_ = [l for l, m in zip(preds[i], mask) if m]
            assert len(labels_) == len(labels)
            ret.append([sentence, labels, labels_])
        return ret

    def predict_on_batch(self, sess, inputs_batch, mask_batch):
        feed = self.create_feed_dict(
            inputs_batch=inputs_batch,
            mask_batch=mask_batch,
            dropout=self.config.dropout)
        predictions = sess.run(tf.argmax(self.pred, axis=2), feed_dict=feed)
        return predictions

    def train_on_batch(self, sess, inputs_batch, labels_batch, mask_batch):
        feed = self.create_feed_dict(inputs_batch,
                                     mask_batch=mask_batch,
                                     labels_batch=labels_batch,
                                     dropout=self.config.dropout)
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss

    def evaluate(self, sess, examples, examples_raw):
        """
        Evaluates model performance on @examples.
        """
        token_cm = ConfusionMatrix(labels=LBLS)

        correct_preds, total_correct, total_preds = 0., 0., 0.
        for _, labels, labels_ in self.output(sess, examples_raw, examples):
            for l, l_ in zip(labels, labels_):
                token_cm.update(l, l_)
            gold = set(get_chunks(labels))
            pred = set(get_chunks(labels_))
            correct_preds += len(gold.intersection(pred))
            total_preds += len(pred)
            total_correct += len(gold)

        p = correct_preds / total_preds if correct_preds > 0 else 0
        r = correct_preds / total_correct if correct_preds > 0 else 0
        f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
        return token_cm, (p, r, f1)

    def output(self, sess, inputs_raw, inputs=None):
        """
        Reports the output of the model on examples (uses helper to featurize each example).
        """
        if inputs is None:
            inputs = self.preprocess_sequence_data(
                self.helper.vectorize(inputs_raw))

        preds = []
        prog = Progbar(target=1 + int(len(inputs) / self.config.batch_size))
        for i, batch in enumerate(minibatches(inputs, self.config.batch_size, shuffle=False)):
            # Ignore predict
            batch = batch[:1] + batch[2:]
            preds_ = self.predict_on_batch(sess, *batch)
            preds += list(preds_)
            prog.update(i + 1, [])
        return self.consolidate_predictions(inputs_raw, inputs, preds)

    def fit(self, sess, saver, train_examples_raw, dev_set_raw):
        best_score = 0.

        train_examples = self.preprocess_sequence_data(train_examples_raw)
        dev_set = self.preprocess_sequence_data(dev_set_raw)

        for epoch in range(self.config.n_epochs):
            logger.info("Epoch %d out of %d", epoch + 1, self.config.n_epochs)
            prog = Progbar(
                target=1 + int(len(train_examples) / self.config.batch_size))

            for i, batch in enumerate(minibatches(train_examples, self.config.batch_size)):
                loss = self.train_on_batch(sess, *batch)
                prog.update(i + 1, [("train loss", loss)])
                if self.report:
                    self.report.log_train_loss(loss)
            print("")

            logger.info("Evaluating on development data")
            token_cm, entity_scores = self.evaluate(sess, dev_set, dev_set_raw)
            logger.debug("Token-level confusion matrix:\n" +
                         token_cm.as_table())
            logger.debug("Token-level scores:\n" + token_cm.summary())
            logger.info("Entity level P/R/F1: %.2f/%.2f/%.2f", *entity_scores)

            score = entity_scores[-1]

            if score > best_score:
                best_score = score
                if saver:
                    logger.info("New best score! Saving model in %s",
                                self.config.model_output)
                    saver.save(sess, self.config.model_output)
            print("")
            if self.report:
                self.report.log_epoch()
                self.report.save()
        return best_score


def pad_sequences(data, max_length):
    """
    Ensures each input-output seqeunce pair in data is of length
    """
    ret = []

    # Use this zero vector when padding sequences.
    zero_vector = [0] * Config.n_features
    zero_label = 4  # corresponds to the 'O' tag

    for sentence, labels in data:
        mask = [True for i in range(max_length)]
        if len(sentence) > max_length:
            filled_sentence = sentence[:max_length]
            filled_labels = labels[:max_length]
        else:
            filled_sentence = sentence.copy()
            filled_labels = labels.copy()
            for i in range(len(sentence), max_length):
                filled_sentence.append(zero_vector)
                filled_labels.append(zero_label)
                mask[i] = False
        ret.append((filled_sentence, filled_labels, mask))
    return ret
