import os.path
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

def input_fn(data_file, col_names, header=None, test_percentage=0.2, label_name='class'):
    """Takes a .csv file and partitions it into test and train data which is
       then returned."""
    try:
        assert os.path.isfile(data_file), ( \
        '%s not found. Please make sure you have run data_download.py and ' \
        'set the --data_dir argument to the correct path.' % data_file)
        print(data_file + " loaded", end='')
    except:
        print("CRITICAL ERROR: " + data_file + " FAILED TO LOAD")
        exit()

    data = pd.read_csv(filepath_or_buffer=data_file,
                           names=col_names,
                           header=None)
    train, test = train_test_split(data, test_size=test_percentage)
    total = len(train) + len(test)
    print(" with {} records.\n".format(total))

    train_features, train_label = train, train.pop(label_name)
    test_features, test_label = test, test.pop(label_name)

    return (train_features, train_label), (test_features, test_label)




# Thanks tensorflow website
def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset

def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features=dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset