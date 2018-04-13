from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

from input_data import input_fn, train_input_fn, eval_input_fn
from csv_class import CSV
from model import model_Adagrad, model_RMSProp

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
_LAYERS = 2

def main(argv):
    # Read in a CSV object
    (train_x, train_y), (test_x, test_y) = input_fn(data_file=csv.name,
                                                    col_names=csv.col_names,
                                                    test_percentage=_TEST_PERCENTAGE,
                                                    label_name=csv.label_name)

    # Feature columns describe how to use the input.
    my_feature_columns = []
    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))
    classifier = tf.estimator.Estimator(
        model_fn=_MODEL,
        params={
            'feature_columns': my_feature_columns,
            # Two hidden layers of 10 nodes each.
            'hidden_units': [_NODES_PER_LAYER, _NODES_PER_LAYER],
            # The model must choose between 3 classes.
            'n_classes': csv.classes,
        })

    # Train the Model.
    classifier.train(
        input_fn=lambda:train_input_fn(train_x, train_y, _BATCH_SIZE),
        steps=csv.num_examples['train'])

    # Evaluate the model.
    eval_result = classifier.evaluate(
        input_fn=lambda:eval_input_fn(test_x, test_y, _BATCH_SIZE))

    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))





if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)