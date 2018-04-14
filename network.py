import tensorflow as tf
from input_data import input_fn, train_input_fn, eval_input_fn
 

def run_network(csv, model, hidden_layers, batch_size, test_percent):
    # Read in a CSV object
    (train_x, train_y), (test_x, test_y) = input_fn(data_file=csv.name,
                                                    col_names=csv.col_names,
                                                    test_percentage=test_percent,
                                                    label_name=csv.label_name)

    # Feature columns describe how to use the input.
    my_feature_columns = []
    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))
    classifier = tf.estimator.Estimator(
        model_fn=model,
        params={
            'feature_columns': my_feature_columns,
            'hidden_units': hidden_layers,
            # The model must choose between csv.classes
            'n_classes': csv.classes,
        })

    # Train the Model.
    with tf.device('/device:GPU:0'):
    # with tf.device('/cpu:0'):
        classifier.train(
            input_fn=lambda:train_input_fn(train_x, train_y, batch_size),
            steps=csv.num_examples['train'])

    # Evaluate the model.
    eval_result = classifier.evaluate(
        input_fn=lambda:eval_input_fn(test_x, test_y, batch_size))
    
    return eval_result