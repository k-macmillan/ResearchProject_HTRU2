from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from math import sqrt

# from input_data import input_fn, train_input_fn, eval_input_fn
from csv_class import CSV
from model import model_Adagrad, model_RMSProp
from network import run_network

_TEST_PERCENTAGE = 0.2
_TRAIN = 10 #int(17898*(1-_TEST_PERCENTAGE))
_BATCH_SIZE = int(_TRAIN*.1)
_MODEL = model_RMSProp

# Create a CSV object
csv = CSV(name='HTRU_2_inverse.csv',
          col_names=['mean_IP', 'std_IP', 'e_kurtosis_IP', 'skewness_IP', 'mean_DM', 'std_DM', 'e_kurtosis_DM', 'skewness_DM', 'class'], 
          label_name='class', 
          col_defaults=[[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0]], 
          num_examples={'train': _TRAIN, 
                        'test': int(_TRAIN*_TEST_PERCENTAGE)}, 
          classes=2)


_NODES_PER_LAYER = int((len(csv.col_defaults) - 1) * 1.5)
_LAYERS = max(int(sqrt(_NODES_PER_LAYER)), 2)
_HIDDEN_LAYERS = [_NODES_PER_LAYER]*_LAYERS

def main(argv):
    # print('\nTest set accuracy: {accuracy:0.3f}\n'.format(run_network(csv, _MODEL, _HIDDEN_LAYERS, _BATCH_SIZE, _TEST_PERCENTAGE)))
    runs = 20
    total_acc = 0.0
    for _ in range(runs):
        total_acc += run_network(csv, _MODEL, _HIDDEN_LAYERS, _BATCH_SIZE, _TEST_PERCENTAGE)['accuracy']

    print("Accuracy = ", total_acc/runs)

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)