
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import shutil
import sys

import tensorflow as tf 

from official.utils.arg_parsers import parsers
from official.utils.logging import hooks_helper

_CSV_COLUMNS = [
    'mean_IP', 'std_IP', 'e_kurtosis_IP', 'skewness_IP', 'mean_DM',
    'std_DM', 'e_kurtosis_DM', 'skewness_DM', 'class'
]

_CSV_COLUMN_DEFAULTS = [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0]]

_NUM_EXAMPLES = {
    'train': 14633,
    'validation': 1626,
}


# https://github.com/tensorflow/models/blob/master/official/wide_deep/wide_deep.py
def input_fn(data_file, num_epochs, shuffle, batch_size):
    """Generate an input function for the Estimator."""
    assert tf.gfile.Exists(data_file), (
        '%s not found. Please make sure you have run data_download.py and '
        'set the --data_dir argument to the correct path.' % data_file)

    def parse_csv(value):
        print('Parsing', data_file)
        columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)
        features = dict(zip(_CSV_COLUMNS, columns))
        labels = features.pop('income_bracket')
        return features, tf.equal(labels, '>50K')

    # Extract lines from input files using the Dataset API.
    dataset = tf.data.TextLineDataset(data_file)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=_NUM_EXAMPLES['train'])

    dataset = dataset.map(parse_csv, num_parallel_calls=5)

    # We call repeat after shuffling, rather than before, to prevent separate
    # epochs from blending together.
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    return dataset


# https://github.com/soerendip/Tensorflow-binary-classification/blob/master/Tensorflow-binary-classification-model.ipynb
def simple_model(in_tensor, num_inputs, num_classes, weights, biases):
    learning_rate = 0.001
    num_epochs = 100
    batch_size = 100  
    num_hidden_1 = 13
    num_hidden_2 = 13


    layer_1 = tf.add(tf.matmul(in_tensor, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)

    out_tensor = 


def main(argv):
    num_inputs = 8
    num_classes = 2
    dataset = input_fn("HTRU_2.csv", num_epochs, False, batch_size)

    ins = tf.placeholder("float", [None, num_inputs])
    outs = tf.placeholder("float", [None, num_classes])

    weights = {
        'h1': tf.Variable(tf.random_normal([num_inputs, num_hidden_1])),
        'h2': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2])),
        'out': tf.Variable(tf.random_normal([num_hidden_2, num_classes]))
    }

    biases = {
        'b1': tf.Variable(tf.random_normal([num_hidden_1])),
        'b2': tf.Variable(tf.random_normal([num_hidden_2])),
        'out': tf.Variable(tf.random_normal([num_classes]))
    }

    predictor = simple_model(ins, num_inputs, num_classes, weights, biases)
    cost_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predictor, labels=outs))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    init = tf.global_variables_initializer()


if __name__ == '__main__':
    # tf.logging.set_verbosity(tf.logging.INFO)
    main(argv=sys.argv)