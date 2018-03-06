#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import sys
import time

import numpy as np
import tensorflow as tf

from data_util import ModelHelper, load_and_preprocess_data, load_embeddings
from defs import LBLS
from model import NERModel, Config
from util import print_sentence, read_conll, write_conll

logger = logging.getLogger("NER")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


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
        model = NERModel(helper, config, embeddings)
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
        model = NERModel(helper, config, embeddings)

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
        model = NERModel(helper, config, embeddings)
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
