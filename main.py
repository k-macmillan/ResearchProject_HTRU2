from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from input_data import input_fn
from csv_class import CSV

_TEST_PERCENTAGE = 0.2

# Create a CSV object
csv = CSV(name='HTRU_2.csv',
          col_names=['mean_IP', 'std_IP', 'e_kurtosis_IP', 'skewness_IP', 'mean_DM', 'std_DM', 'e_kurtosis_DM', 'skewness_DM', 'class'], 
          label_name='class', 
          col_defaults=[[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0]], 
          num_examples={'train': int(17898*(1-_TEST_PERCENTAGE)), 
                        'test': int(17898*_TEST_PERCENTAGE)}, 
          classes=2)


def main():
    # Read in a CSV object
    (train_x, train_y), (test_x, test_y) = input_fn(data_file=csv.name,
                                                    col_names=csv.col_names,
                                                    test_percentage=_TEST_PERCENTAGE,
                                                    label_name=csv.label_name)








if __name__ == '__main__':
    main()