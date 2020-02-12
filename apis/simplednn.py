import adanet
import tensorflow as tf
import functools

FEATURES_KEY = 'x'
_NUM_LAYERS_KEY = "num_layers"


class _SimpleDNNBuilder(adanet.subnetwork.Builder):

    def __init__(self, optimizer, layer_size, num_layers, learn_mixture_weights, seed):

        self._optimizer = optimizer
        self._layer_size = layer_size
        self._num_layers = num_layers
        self._learn_mixture_weights = learn_mixture_weights
        self._seed = seed

    def build_subnetwork(self,
                         features,
                         logits_dimension,
                         training,
                         iteration_step,
                         summary,
                         previous_ensemble=None):

        input_layer = tf.to_float(features[FEATURES_KEY])
        kernel_initializer = tf.glorot_uniform_initializer(seed=self._seed)
        last_layer = input_layer

        # hidden layesr
        for _ in range(self._num_layers):
            last_layer = tf.layers.dense(
                last_layer,
                units=self._layer_size,
                activation=tf.nn.relu,
                kernel_initializer=kernel_initializer)

        # logits are input to activation function
        logits = tf.layers.dense(
            last_layer,
            units=logits_dimension,
            kernel_initializer=kernel_initializer)

        # At the end of iteration,the tf.Tensor instances (hidden layers) will be available to subnetworks in the next iterations
        persisted_tensors = {_NUM_LAYERS_KEY: tf.constant(self._num_layers)}

        return adanet.Subnetwork(
            last_layer=last_layer,
            logits=logits,
            complexity=self._measure_complexity(),
            persisted_tensors=persisted_tensors)

    def _measure_complexity(self):
        """Approximates complexity as the square-root of the depth."""
        return tf.sqrt(tf.to_float(self._num_layers))

    def build_subnetwork_train_op(self, subnetwork, loss, var_list, labels, iteration_step, summary, previous_ensemble):
        return self._optimizer.minimize(loss=loss, var_list=var_list)

    def build_mixture_weights_train_op(self, loss, var_list, logits, labels, iteration_step, summary):
        if not self._learn_mixture_weights:
            return tf.no_op("mixture_weights_train_op")
        # Allows Adanet to learn mixture weights of each subnetwork on its own
        return self._optimizer.minimize(loss=loss, var_list=var_list)

    # Function returns subnetwork's name as no of hidden layers+ _layer_dnn
    @property
    def name(self):
        if self._num_layers == 0:
            # A DNN with no hidden layers is a linear model.
            return "linear"
        return "{}_layer_dnn".format(self._num_layers)


class SimpleDNNGenerator(adanet.subnetwork.Generator):
    def __init__(self,
                 optimizer,
                 layer_size,
                 learn_mixture_weights=False,
                 seed=None):

        self._seed = seed
        self._dnn_builder_fn = functools.partial(
            _SimpleDNNBuilder,
            optimizer=optimizer,
            layer_size=layer_size,
            learn_mixture_weights=learn_mixture_weights)

    def generate_candidates(self, previous_ensemble, iteration_number, previous_ensemble_reports, all_reports):
        num_layers = 1
        seed = self._seed

        if previous_ensemble:
            # initializing num_layers to persisted_tensors(hidden_layers) to be used in current iteration
            num_layers = tf.contrib.util.constant_value(
                previous_ensemble.weighted_subnetworks[-1].subnetwork.persisted_tensors[_NUM_LAYERS_KEY])

        # Change seed to make subnetwork learns different in each iteration
        if seed is not None:
            seed += iteration_number
        return [
            self._dnn_builder_fn(num_layers=num_layers, seed=seed),
            self._dnn_builder_fn(num_layers=num_layers + 1, seed=seed)
        ]
