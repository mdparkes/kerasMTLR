# kerasMTLR
An implementation of the MTLR algorithm for survival prediction using TensorFlow with Keras. Based on the paper ["Learning Patient-Specific Cancer Survival Distributions as a Sequence of Dependent Regressors"](http://www.cs.cornell.edu/~cnyu/papers/nips11_survival.pdf) by Yu *et al* (NIPS 2011), but does not use the second parameter smoothing regularizer, as it has been shown that this regularizer is unnecessary.

## Details
The `NeuralNetworkMTLRSurvival` class extends `keras.Model` and allows the user to model individualized survival distributions from a latent representation of observed input features using MTLR. See below for a more detailed description of the MTLR algorithm. This model works with right-censored data, but does not support left- or interval-censored data. Its forward pass outputs a cumulative distribution of probabilities that a patient will survive at least until the beginning of each of a series of time intervals modeled by MTLR.

The `NeuralNetworkMTLRSurvival` `Model` can be initialized with a callable `keras.Layer` object which calculates a latent representation of the input examples' observable features. The latent representation is used as input to the final `MTLR` `Layer` of the model, and is learned in a supervised, end-to-end fashion using the loss minimization objective of the `MTLR` `Layer`. Specifically, the objective is to find model parameters that maximize the likelihood (minimize the negative log likelihood) of observing the training survival data.

Note that Keras' own `Sequential` class is not compatible with `NeuralNetworkMTLRSurvival` because of how the latter calculates the loss from its inputs. If the user wishes to use a sequential block of Keras `Layer`s to calculate a latent representation of their data, they can use the `SequentialBlock` `Layer` subclass in `layers.py`. `SequentialBlock.__init__` takes a list of Keras `Layer`s and constructs a sequential block with the `Layer`s stacked in the order they appear in the list.

Supplying a `keras.Layer` to the `NeuralNetworkMTLRSurvival` initializer is optional. If the user does not provide an upstream `Layer` that learns a latent representation for input to the MTLR output `Layer`, the MTLR output `Layer` will simply use the features of the data records as input. This is useful if the user wants to supply a previously learned unsupervised representation of their data as input to MTLR, or if they simply want to use the observed features.

## MTLR Details
Time (measured in units of the training data) is partitioned into m+1 right-open intervals that span 0 to infinity. For each patient we can construct a monotonically increasing survival sequence of m+1 bits which expresses all the intervals through which the patient survived (denoted by a value of 0) and all the intervals that the patient did not live through (denoted by a value of 1). Once a value of 1 is assigned, all bits thereafter must also be 1 (once dead, we tend to stay dead). For example, if we partition time into 5 intervals (i.e. m=4) and a patient's time of death was in the third interval, their survival sequence would be [0, 0, 1, 1, 1].

The trainable parameters of the MTLR layer maximize the likelihood of the observed times of death or censoring in the training dataset, given the associated input features. If a patient is censored (i.e. we only know the lower bound on their time of death, but not their true time of death), MTLR models the likelihood by marginalizing over all possible survival sequences that are compatible with the censoring time (the lower bound on time of death). If the patient is uncensored (the time of death is known), MTLR models the likelihood by considering the one survival sequence that represents the patient's known time of death.

The `MTLR` `Layer`'s forward pass returns an array of probabilities that a patient's time of death was greater than or equal to the lower bound of each of the intervals that were used to partition time. The probability of an event occurring in any particular interval can be calculated by a forward pass of the `NeuralNetworkMTLRSurvival.mtlr` `Layer`.

## Usage Example
```
import numpy as np
import tensorflow as tf

from layers import SequentialBlock
from models import NeuralNetworkMTLRSurvival


if __name__ == "__main__":
    # Simulated survival data
    time_labels = tf.cast(tf.random.uniform(shape=(100, ), minval=0, maxval=862, dtype=tf.int32, seed=42), tf.float32)
    event_labels = tf.cast(tf.random.uniform(shape=(100, ),  minval=0, maxval=2, dtype=tf.int32, seed=42), tf.float32)
    features = tf.random.normal(shape=(100, 25), mean=4.6, stddev=1.3, dtype=tf.float32, seed=42)

    # Time intervals for MTLR
    n_intervals = 5
    quantiles = np.linspace(0, 1, n_intervals, endpoint=False)[1:]
    upper_limits = np.append(np.quantile(time_labels.numpy(), quantiles), np.inf)

    # Create datasets
    train_time_labels, val_time_labels, test_time_labels = tf.split(time_labels, [60, 20, 20], 0)
    train_event_labels, val_event_labels, test_event_labels = tf.split(event_labels, [60, 20, 20], 0)
    train_features, val_features, test_features = tf.split(features, [60, 20, 20], 0)

    ds_train = tf.data.Dataset.from_tensor_slices((train_features, train_time_labels, train_event_labels))
    ds_val = tf.data.Dataset.from_tensor_slices((val_features, val_time_labels, val_event_labels))
    ds_test = tf.data.Dataset.from_tensor_slices((test_features, test_time_labels, test_event_labels))

    ds_train_batched = ds_train.repeat().batch(10)  # Allows steps_per_epoch to be set during training
    ds_val_batched = ds_val.batch(10)
    ds_test_batched = ds_test.batch(10)

    # A two-layer MLP is used to calculate the latent representation for input to MTLR
    dense_1 = tf.keras.layers.Dense(5, name="mlp_layer_1")
    dense_2 = tf.keras.layers.Dense(7, name="mlp_layer_2")
    nn_block = SequentialBlock([dense_1, dense_2], name="mlp_block")
    model = NeuralNetworkMTLRSurvival(nn_block=nn_block, interval_upper=upper_limits)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), weighted_metrics=[])
    history = model.fit(ds_train_batched, steps_per_epoch=6, epochs=5, validation_data=ds_val_batched)
    model.evaluate(ds_test_batched)
```

