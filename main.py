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
_BATCH_SIZE = 100
_MODEL = model_RMSProp

# Create a CSV object
csv = CSV(name='HTRU_2.csv',
          col_names=['mean_IP', 'std_IP', 'e_kurtosis_IP', 'skewness_IP', 'mean_DM', 'std_DM', 'e_kurtosis_DM', 'skewness_DM', 'class'], 
          label_name='class', 
          col_defaults=[[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0]], 
          num_examples={'train': int(17898*(1-_TEST_PERCENTAGE)), 
                        'test': int(17898*_TEST_PERCENTAGE)}, 
          classes=2)


_NODES_PER_LAYER = int((len(csv.col_defaults) - 1) * 1.5)
_LAYERS = max(int(sqrt(_NODES_PER_LAYER)), 2)
_HIDDEN_LAYERS = [_NODES_PER_LAYER]*_LAYERS

def main(argv):
    run_network(csv, _MODEL, _HIDDEN_LAYERS, _BATCH_SIZE, _TEST_PERCENTAGE)

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)