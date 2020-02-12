import tensorflow as tf
import adanet
from simplednn import SimpleDNNGenerator
from adanet_helpers import adanet_helper as ah

import warnings
warnings.filterwarnings("ignore")


class AdanetModelBuilding:

    # Function that implements AdaNet algorithm using its estimator. It trains and evaluates the model iteratively
    @staticmethod
    def train_and_evaluate(output_neurons, LEARNING_RATE, learn_mixture_weights, adanet_lambda, max_iteration_steps,
                           TRAIN_STEPS, BATCH_SIZE, RANDOM_SEED, train_test, model_dir):
        AdaNetEstimator = adanet.Estimator(
            # head instance computes loss and evaluation metrics for every candidate
            head=ah.head_classify(output_neurons),
            # defines candidate subnetworks to train and evaluate at every AdaNet iteration
            subnetwork_generator=SimpleDNNGenerator(
                optimizer=tf.train.AdamOptimizer(learning_rate=LEARNING_RATE),  # adagrad
                layer_size=ah.hidden_layer_neurons(2, train_test),
                learn_mixture_weights=learn_mixture_weights,
                seed=RANDOM_SEED),
            adanet_lambda=adanet_lambda,
            max_iteration_steps=max_iteration_steps,

            # reports are made available to subnetwork_generator in next iteration report_materializer=
            # adanet.ReportMaterializer(input_fn=ah.input_fn("train", training=False, batch_size=BATCH_SIZE,
            # train_test=train_test)),

            config=tf.estimator.RunConfig(
                # save_checkpoints_steps=50000,
                # save_summary_steps=50000,
                tf_random_seed=RANDOM_SEED,
                model_dir=model_dir
            )
        )

        # Determines input data for training
        train_spec = tf.estimator.TrainSpec(
            input_fn=ah.input_fn("train", training=True, batch_size=BATCH_SIZE, train_test=train_test),
            max_steps=TRAIN_STEPS)

        # Combines evaluation metrics of trained models
        eval_spec = tf.estimator.EvalSpec(
            input_fn=ah.input_fn("test", training=False, batch_size=BATCH_SIZE, train_test=train_test),
            steps=None)

        return tf.estimator.train_and_evaluate(AdaNetEstimator, train_spec, eval_spec)
