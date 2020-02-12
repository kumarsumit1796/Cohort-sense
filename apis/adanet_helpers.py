import tensorflow as tf

import warnings
warnings.filterwarnings("ignore")
FEATURES_KEY = 'x'


class adanet_helper:
    # Function to provide input data to AdaNet Estimator
    # ,dtype=tf.int32
    @staticmethod
    def input_fn(partition, training, batch_size, train_test):
        """Generate an input function for the Estimator."""
        def _input_fn():
            if partition == "train":
                dataset = tf.data.Dataset.from_tensor_slices(({
                                                                  FEATURES_KEY: tf.convert_to_tensor(train_test[0])
                                                              }, tf.convert_to_tensor(train_test[2])))
            else:
                dataset = tf.data.Dataset.from_tensor_slices(({
                                                                  FEATURES_KEY: tf.convert_to_tensor(train_test[1])
                                                              }, tf.convert_to_tensor(train_test[3])))

            dataset = dataset.batch(batch_size)
            iterator = dataset.make_one_shot_iterator()
            features, labels = iterator.get_next()
            return features, labels
        return _input_fn

    # Function to automate neurons in hidden layer with respect to input features
    @staticmethod
    def hidden_layer_neurons(output_neurons, train_test):
        input_neurons = list(train_test[0].shape)[1]
        print("HIDDEN_NEURONS:", int((2 / 3) * input_neurons + output_neurons))
        print("Input:", input_neurons)
        return int((2 / 3) * input_neurons + output_neurons)

    # Function that automates head variable with respect to Output_Neurons in an output layer
    @staticmethod
    def head_classify(output_neurons):
        if output_neurons == 1:
            head = tf.contrib.estimator.regression_head()
            print("=============REGRESSION=========")
        elif output_neurons == 2:
            head = tf.contrib.estimator.binary_classification_head()  # binary_cross_entropy  loss
            print("=============BINARY_CLASSIFICATION=========")
        else:
            head = tf.contrib.estimator.multi_class_head(n_classes=output_neurons)
            print("=============MULTI_CLASS_CLASSIFICATION=========")
        return head
