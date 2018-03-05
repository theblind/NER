#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q2: Recurrent neural nets for NER
"""


import argparse
import logging
import sys
import time
from datetime import datetime

import tensorflow as tf
import numpy as np

from util import print_sentence, write_conll, read_conll
from data_util import load_and_preprocess_data, load_embeddings, ModelHelper
from ner_model import NERModel
from defs import LBLS

logger = logging.getLogger("NER")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


class Config:
    """
    Holds model hyperparams and data information.
    """
    # Number of features for every word in the input
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
    n_epochs = 10
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


class RNNModel(NERModel):

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
        Adds the unrolled RNN
        """

        x = self.add_embedding()
        dropout_rate = self.dropout_placeholder

        preds = []  # Predicted output at each timestep should go here!

        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.config.hidden_size)
        lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell,
                                                  input_keep_prob=1-self.config.dropout,
                                                  output_keep_prob=1-self.config.dropout)

        outputs, state = tf.nn.dynamic_rnn(lstm_cell, x, dtype=tf.float32)
        preds = tf.contrib.layers.fully_connected(
            outputs, self.config.n_classes, activation_fn=None)

        return preds

    def add_loss_op(self, preds):
        """
        Adds Ops for the loss function to the computational graph.
        """
        # seq_length = tf.reduce_sum(tf.cast(self.mask_placeholder, tf.int32))
        # log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(
        #     inputs=tf.boolean_mask(preds, self.mask_placeholder),
        #     tag_indices=tf.boolean_mask(self.labels_placeholder, self.mask_placeholder),
        #     sequence_lengths=seq_length
        # )
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

    def __init__(self, helper, config, pretrained_embeddings, report=None):
        super(RNNModel, self).__init__(helper, config, report)
        self.max_length = min(Config.max_length, helper.max_length)
        # Just in case people make a mistake.
        Config.max_length = self.max_length
        self.pretrained_embeddings = pretrained_embeddings

        # Defining placeholders.
        self.input_placeholder = None
        self.labels_placeholder = None
        self.mask_placeholder = None
        self.dropout_placeholder = None

        self.build()


def do_train(args):
    # Set up some parameters.
    config = Config(args)
    helper, train, dev, train_raw, dev_raw = load_and_preprocess_data(args)
    embeddings = load_embeddings(args, helper)
    config.embed_size = embeddings.shape[1]
    helper.save(config.output_path)

    handler = logging.FileHandler(config.log_output)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(
        '%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)

    report = None  # Report(Config.eval_output)

    with tf.Graph().as_default():
        logger.info("Building model...",)
        start = time.time()
        model = RNNModel(helper, config, embeddings)
        logger.info("took %.2f seconds", time.time() - start)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as session:
            session.run(init)
            model.fit(session, saver, train, dev)
            if report:
                report.log_output(model.output(session, dev_raw))
                report.save()
            else:
                # Save predictions in a text file.
                output = model.output(session, dev_raw)
                sentences, labels, predictions = list(zip(*output))
                predictions = [[LBLS[l] for l in preds]
                               for preds in predictions]
                output = list(zip(sentences, labels, predictions))

                with open(model.config.conll_output, 'w') as f:
                    write_conll(f, output)
                with open(model.config.eval_output, 'w') as f:
                    for sentence, labels, predictions in output:
                        print_sentence(f, sentence, labels, predictions)


def do_evaluate(args):
    config = Config(args)
    helper = ModelHelper.load(args.model_path)
    input_data = read_conll(args.data)
    embeddings = load_embeddings(args, helper)
    config.embed_size = embeddings.shape[1]

    with tf.Graph().as_default():
        logger.info("Building model...",)
        start = time.time()
        model = RNNModel(helper, config, embeddings)

        logger.info("took %.2f seconds", time.time() - start)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as session:
            session.run(init)
            saver.restore(session, model.config.model_output)
            for sentence, labels, predictions in model.output(session, input_data):
                predictions = [LBLS[l] for l in predictions]
                print_sentence(args.output, sentence, labels, predictions)


def do_shell(args):
    config = Config(args)
    helper = ModelHelper.load(args.model_path)
    embeddings = load_embeddings(args, helper)
    config.embed_size = embeddings.shape[1]

    with tf.Graph().as_default():
        logger.info("Building model...",)
        start = time.time()
        model = RNNModel(helper, config, embeddings)
        logger.info("took %.2f seconds", time.time() - start)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as session:
            session.run(init)
            saver.restore(session, model.config.model_output)

            while True:
                # Create simple REPL
                try:
                    sentence = eval(input("input> "))
                    tokens = sentence.strip().split(" ")
                    for sentence, _, predictions in model.output(session, [(tokens, ["O"] * len(tokens))]):
                        predictions = [LBLS[l] for l in predictions]
                        print_sentence(sys.stdout, sentence, [
                                       ""] * len(tokens), predictions)
                except EOFError:
                    print("Closing session.")
                    break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Trains and tests an NER model')
    subparsers = parser.add_subparsers()

    command_parser = subparsers.add_parser('train', help='')
    command_parser.add_argument('-dt', '--data-train', type=argparse.FileType(
        'r'), default="data/train.conll", help="Training data")
    command_parser.add_argument(
        '-dd', '--data-dev', type=argparse.FileType('r'), default="data/dev.conll", help="Dev data")
    command_parser.add_argument('-v', '--vocab', type=argparse.FileType(
        'r'), default="data/vocab.txt", help="Path to vocabulary file")
    command_parser.add_argument('-vv', '--vectors', type=argparse.FileType(
        'r'), default="data/wordVectors.txt", help="Path to word vectors file")
    command_parser.add_argument(
        '-c', '--cell', choices=["rnn", "gru"], default="rnn", help="Type of RNN cell to use.")
    command_parser.set_defaults(func=do_train)

    command_parser = subparsers.add_parser('evaluate', help='')
    command_parser.add_argument(
        '-d', '--data', type=argparse.FileType('r'), default="data/dev.conll", help="Training data")
    command_parser.add_argument('-m', '--model-path', help="Training data")
    command_parser.add_argument('-v', '--vocab', type=argparse.FileType(
        'r'), default="data/vocab.txt", help="Path to vocabulary file")
    command_parser.add_argument('-vv', '--vectors', type=argparse.FileType(
        'r'), default="data/wordVectors.txt", help="Path to word vectors file")
    command_parser.add_argument(
        '-c', '--cell', choices=["rnn", "gru"], default="rnn", help="Type of RNN cell to use.")
    command_parser.add_argument(
        '-o', '--output', type=argparse.FileType('w'), default=sys.stdout, help="Training data")
    command_parser.set_defaults(func=do_evaluate)

    command_parser = subparsers.add_parser('shell', help='')
    command_parser.add_argument('-m', '--model-path', help="Training data")
    command_parser.add_argument('-v', '--vocab', type=argparse.FileType(
        'r'), default="data/vocab.txt", help="Path to vocabulary file")
    command_parser.add_argument('-vv', '--vectors', type=argparse.FileType(
        'r'), default="data/wordVectors.txt", help="Path to word vectors file")
    command_parser.add_argument(
        '-c', '--cell', choices=["rnn", "gru"], default="rnn", help="Type of RNN cell to use.")
    command_parser.set_defaults(func=do_shell)

    ARGS = parser.parse_args()
    if ARGS.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        ARGS.func(ARGS)
