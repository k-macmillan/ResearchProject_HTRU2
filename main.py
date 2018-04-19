from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from math import sqrt
import sys

# from input_data import input_fn, train_input_fn, eval_input_fn
from csv_class import CSV
from model import model_Adagrad, model_RMSProp
from network import run_network

_TEST_PERCENTAGE = 0.2
_TRAIN = int(17898*(1-_TEST_PERCENTAGE))
_BATCH_SIZE = max(int(_TRAIN*.01), 1)
_MODEL = model_RMSProp

# Create a CSV object
csv = CSV(name='HTRU_2.csv',
          col_names=['mean_IP', 'std_IP', 'e_kurtosis_IP', 'skewness_IP', 'mean_DM', 'std_DM', 'e_kurtosis_DM', 'skewness_DM', 'class'], 
          label_name='class', 
          col_defaults=[[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0]], 
          num_examples={'train': _TRAIN, 
                        'test': int(_TRAIN*_TEST_PERCENTAGE)}, 
          classes=2)

# Based on CSV file:
_NODES_PER_LAYER = int((len(csv.col_defaults) - 1) * 1.5)
_LAYERS = max(int(sqrt(_NODES_PER_LAYER)), 2)
_HIDDEN_LAYERS = [_NODES_PER_LAYER]*_LAYERS

def main(argv):
    runs = 1
    try:
        if len(argv) > 1:
            runs = int(argv[1])        
    except ValueError:
        print("Please specify an integer amount of runs or nothing at all")
        exit()
    
    total_acc = 0.0
    # Allows for multiple runs to acquire an average
    for run in range(runs):
        print("Run: ", run)
        total_acc += run_network(csv, _MODEL, _HIDDEN_LAYERS, _BATCH_SIZE, _TEST_PERCENTAGE, run)['accuracy']


    print("Accuracy = ", total_acc / runs)

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)