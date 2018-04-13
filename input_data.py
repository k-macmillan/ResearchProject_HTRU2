import os.path
import pandas as pd
from sklearn.model_selection import train_test_split

def input_fn(data_file, col_names, header=None, test_size=0.2):
    """Generate an input function for the Estimator."""
    try:
        assert os.path.isfile(data_file), ( \
        '%s not found. Please make sure you have run data_download.py and ' \
        'set the --data_dir argument to the correct path.' % data_file)
        print(data_file + " loaded")
    except:
        print("CRITICAL ERROR: " + data_file + " FAILED TO LOAD")
        exit()

    data = pd.read_csv(filepath_or_buffer=data_file,
                           names=col_names,
                           header=None)
    train, test = train_test_split(data, test_size=0.2)

    label_name = 'class'
    train_features, train_label = train, train.pop(label_name)
    test_features, test_label = test, test.pop(label_name)

    return (train_features, train_label), (test_features, test_label)