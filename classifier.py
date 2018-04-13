
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import shutil
import sys

import tensorflow as tf 
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np


from input_data import input_fn

# from official.utils.arg_parsers import parsers
# from official.utils.logging import hooks_helper

_CSV_COLUMN_NAMES = [
    'mean_IP', 'std_IP', 'e_kurtosis_IP', 'skewness_IP', 'mean_DM',
    'std_DM', 'e_kurtosis_DM', 'skewness_DM', 'class'
]

_CSV_COLUMN_DEFAULTS = [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0]]

_NUM_EXAMPLES = {
    'train': int(17898*.8),
    'test': int(17898*.2) + 1,
}



# https://github.com/soerendip/Tensorflow-binary-classification/blob/master/Tensorflow-binary-classification-model.ipynb
def simple_model(in_tensor, num_inputs, num_classes, weights, biases):
    learning_rate = 0.001
    num_epochs = 100
    batch_size = 100  
    num_hidden_1 = 10
    num_hidden_2 = 10


    layer_1 = tf.add(tf.matmul(in_tensor, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)

    out_tensor = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_tensor


def main(argv):



    learning_rate = 0.001
    num_epochs = 100
    batch_size = 100  
    num_hidden_1 = 10
    num_hidden_2 = 10


    (train_feature, train_label), (test_feature, test_label) = input_fn("HTRU_2.csv", _CSV_COLUMN_NAMES)
    print("Data loaded!")
    exit()

    X, Y = make_classification(n_samples=50000, n_features=10, n_informative=8, 
                           n_redundant=0, n_clusters_per_class=2)
    Y = np.array([Y, -(Y-1)]).T  # The model currently needs one column for each class
    X, X_test, Y, Y_test = train_test_split(X, Y)

    num_inputs = 10
    num_classes = 2
    # dataset = input_fn("HTRU_2.csv", num_epochs, False, batch_size)

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
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost_func)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(num_epochs):
            avg_cost = 0.
            total_batch = int(len(X)/batch_size)
            X_batches = np.array_split(X, total_batch)
            Y_batches = np.array_split(Y, total_batch)
            # Loop over all batches
            for i in range(total_batch):
                batch_x, batch_y = X_batches[i], Y_batches[i]
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([optimizer, cost_func], feed_dict={ins: batch_x,
                                                              outs: batch_y})
                # Compute average loss
                avg_cost += c / total_batch
            # Display logs per epoch step
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

        print("Optimization Finished!")

        # Test model
        correct_prediction = tf.equal(tf.argmax(predictor, 1), tf.argmax(outs, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Accuracy:", accuracy.eval({ins: X_test, outs: Y_test}))
        global result 
        result = tf.argmax(predictor, 1).eval({ins: X_test, outs: Y_test})

if __name__ == '__main__':
    # tf.logging.set_verbosity(tf.logging.INFO)
    main(argv=sys.argv)