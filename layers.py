"""
MIT License
---
Copyright (c) 2023 Michael Parkes

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
---
"""
import numpy as np
import tensorflow as tf

from tensorflow import Tensor
from typing import Any, Dict, List, Optional, Union


ArrayLike = Union[List, np.ndarray, Tensor]


class SequentialBlock(tf.keras.layers.Layer):
    """
    Create a block of sequentially connected Keras `Layer`s. Keras' own Sequential models do not play well with the
    `NeuralNetworkMTLRSurvival` `Model` class; this is a simple alternative that will work.

    During initialization, each component `Layer` is stored as an attribute accessed as the `Layer`'s own `name`,
    so it is advised to supply component `Layer`s with unambiguous `name's before initializing the `SequentialBlock`.
    The component `Layer` names are stored as a list under `self.layer_names`.
    """

    def __init__(self, layers: List, **kwargs):
        """
        Initialize the `SequentialBlock` with `Layer`s stacked in the order presented in the `layers` argument.

        :param layers: a list of `tf.keras.layers.Layer` objects
        """
        kwargs.setdefault("name", "sequential_block")
        super().__init__(**kwargs)
        self.layer_names = []
        for layer in layers:
            setattr(self, f"{layer.name}", layer)
            self.layer_names.append(layer.name)

    def get_config(self) -> Dict[str, Any]:
        config = super(SequentialBlock, self).get_config()
        config.update(
            {"layers": [getattr(self, name) for name in self.layer_names]}
        )
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        return cls(**config)

    def call(self, inputs, *args, **kwargs):
        next_input = inputs
        for layer in self.layer_names:
            next_input = getattr(self, layer)(next_input)
        return next_input


class MTLR(tf.keras.layers.Layer):
    """
    This layer stores trainable parameters for survival modeling by multi-task logistic regression. Given some
    input features, its forward pass predicts a patient's probability of surviving to each of time points fitted by
    the MTLR model.

    Details:

    Time (measured in units of the training data) is partitioned into m+1 right-open intervals that span 0 to
    infinity. For each patient we can construct a monotonically increasing survival sequence of m+1 bits which
    expresses all the intervals through which the patient survived (denoted by a value of 0) and all the intervals
    that the patient did not live through (denoted by a value of 1). Once a value of 1 is assigned,
    all bits thereafter must also be 1 (once dead, we tend to stay dead). For example, if we partition time into 5
    intervals (i.e. m=4) and a patient's time of death was in the third interval, their survival sequence would be
    [0, 0, 1, 1, 1].

    The trainable parameters of the MTLR layer maximize the likelihood of the observed times of death or censoring
    in the training dataset, given the associated input features. If a patient is censored (i.e. we only know the lower
    bound on their time of death, but not their true time of death), MTLR models the likelihood by marginalizing over
    all possible survival sequences that are compatible with the censoring time (the lower bound on time of death).
    If the patient is uncensored (the time of death is known), MTLR models the likelihood by considering the one
    survival sequence that represents the patient's known time of death.

    The layer's forward pass returns an array of probabilities that a patient's time of death was in a particular
    interval.
    """
    def __init__(self,
                 kernel_initializer: Union[str, tf.keras.initializers.Initializer] = "random_normal",
                 bias_initializer: Union[str, tf.keras.initializers.Initializer] = "zeros",
                 kernel_regularizer: Union[None, tf.keras.regularizers.Regularizer] = None,
                 bias_regularizer: Union[None, tf.keras.regularizers.Regularizer] = None,
                 *,
                 interval_lower: Optional[ArrayLike] = None,
                 interval_upper: Optional[ArrayLike] = None,
                 **kwargs):
        """
        :param kernel_initializer: Either a Keras Initializer or a string that Keras accepts as shorthand for the
        desired initializer
        :param bias_initializer: Either a Keras Initializer or a string that Keras accepts as shorthand for the
        desired initializer
        :param kernel_regularizer: A Keras Regularizer
        :param bias_regularizer: A Keras Regularizer
        :param interval_lower: An array-like sequence of lower bounds on intervals that fully partition time. One of
        interval_lower or interval_upper must be provided.
        :param interval_upper: An array-like sequence of upper bounds on intervals that fully partition time. One of
        interval_lower or interval_upper must be provided.
        :param kwargs: Keyword arguments passed to keras.layer.Layer's __init__ method
        """
        # Check for interval arguments and convert to Tensor if necessary
        if interval_lower is None and interval_upper is None:
            raise SyntaxError("One of 'interval_lower' or 'interval_upper' must be supplied to __init__")
        # Create interval tensors if unspecified in args to __init__
        if interval_lower is not None:
            interval_lower = tf.constant(interval_lower) if not isinstance(interval_lower, Tensor) else interval_lower
            if interval_upper is None:
                interval_upper = tf.concat([interval_lower[1:], [np.inf]], axis=0)
        if interval_upper is not None:
            interval_upper = tf.constant(interval_upper) if not isinstance(interval_upper, Tensor) else interval_upper
            if interval_lower is None:
                interval_lower = tf.concat([[0], interval_upper[:-1]], axis=0)

        kwargs.setdefault("name", "mtlr")
        # Include a default input_spec in kwargs? -> see docs on InputSpec objects.
        super(MTLR, self).__init__(**kwargs)
        self.n_intervals = len(interval_lower)
        self.n_times = self.n_intervals - 1  # m
        self.interval_lower = tf.cast(interval_lower, tf.float32)
        self.interval_upper = tf.cast(interval_upper, tf.float32)
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        # Create a triangular matrix that will be used in matrix multiplication to sum scores over valid survival
        # sequences given the censoring status and time.
        tri = tf.ones((self.n_times, self.n_times))
        ltri = tf.linalg.LinearOperatorLowerTriangular(tri).to_dense()
        self.ltri = tf.concat([ltri, tf.zeros((self.n_times, 1))], axis=1)

    def get_config(self):
        config = super(MTLR, self).get_config()
        interval_lower = self.interval_lower.numpy().tolist()  # Make JSON-compatible for serialization
        interval_upper = self.interval_upper.numpy().tolist()  # Make JSON-compatible for serialization
        config.update(
            {"kernel_initializer": self.kernel_initializer, "bias_initializer": self.bias_initializer,
             "kernel_regularizer": self.kernel_regularizer, "bias_regularizer": self.bias_regularizer,
             "interval_lower": interval_lower, "interval_upper": interval_upper}
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def build(self, input_shape):
        """Initialize the MTLR layer's weights and biases"""
        # One weight vector for each of the first m intervals
        # The trainable weights need to be initialized with a name attribute because there is a bug that interferes
        # with model checkpointing if they are not. This bug affects hyperparameter tuning with keras-tuner. See
        # https://github.com/tensorflow/tensorflow/issues/26811#issuecomment-474255444
        self.w = self.add_weight(name="w", shape=(input_shape[-1], self.n_times),
                                 initializer=self.kernel_initializer,
                                 regularizer=self.kernel_regularizer,
                                 trainable=True)
        # One bias value for each of the first m intervals
        self.b = self.add_weight(name="b", shape=(self.n_times, ),
                                 initializer=self.bias_initializer,
                                 regularizer=self.bias_regularizer,
                                 trainable=True)

    @tf.function
    def surv_seq(self, time: Tensor, censor: Tensor) -> Tensor:
        """
        Creates a binary vector that indicates the time intervals by which an event occurred (if uncensored) or the
        intervals where death could have occurred (if censored). For censored data, the intervals are
        [t[0]=0, t[1]), ..., [t[j-1], t[j]), ..., [t[J], t[J+1]=Inf). The jth component of the binary vector has a
        value of 1 if the event occurred or could have occurred in the interval [t[j], t[j+1]). Otherwise, the value is
        0. In the uncensored case, if the patient died in jth interval, all components from j through J are 1,
        and all components from 0 through j-1 are 0.

        :param time: The observed time of event or censoring
        :param censor: Indicates whether time is left-censored (-1), right-censored (0), or uncensored (1)
        :return: A binary sequence indicating intervals by which the event had occurred (uncensored) or where it may
        have occurred (censored)
        """

        def tensor_as_vector(tensor):
            """If `tensor` is a scalar, make it a vector. If it is already a vector, leave it as-is."""
            vector = tf.case(
                [(tf.equal(tf.rank(tensor), 0),
                  lambda: tf.reshape(tensor, shape=(-1,)))],
                default=lambda: tensor
            )
            return vector

        time = tensor_as_vector(time)
        # Dynamic partitioning can only use non-negative integers, so left-censoring values of -1 are substituted with 2
        censor = tf.where(censor == -1, tf.fill(tf.shape(censor), 2), tf.cast(censor, tf.int32))
        obs_idx = tf.range(tf.shape(time)[0])  # Indices of each time in the data set
        condition_indices = tf.dynamic_partition(obs_idx, censor, 3)  # For restitching
        # rc: right censored (censor = 0); uc: uncensored (censor = 1); lc: left censored (censor = 2)
        rc_time, uc_time, lc_time = tf.dynamic_partition(time, censor, 3)  # List of partitions
        rc_seq = tf.map_fn(lambda t: tf.where(t < self.interval_upper,
                                              tf.ones_like(self.interval_upper),
                                              tf.zeros_like(self.interval_upper)), rc_time)
        uc_seq = tf.map_fn(lambda t: tf.where(t < self.interval_upper,
                                              tf.ones_like(self.interval_upper),
                                              tf.zeros_like(self.interval_upper)), uc_time)
        lc_seq = tf.map_fn(lambda t: tf.where(t >= self.interval_lower,
                                              tf.ones_like(self.interval_upper),
                                              tf.zeros_like(self.interval_upper)), lc_time)
        sequence = tf.dynamic_stitch(condition_indices, data=[rc_seq, uc_seq, lc_seq])

        return sequence

    @tf.function
    def survival_fn(self, features: Tensor, time: Tensor) -> Tensor:
        """Calculates the probability of observing the survival sequence(s) implied by death at the specified time"""
        # Below we specify the `event` tensor as a tensor full of ones for use with `self.surv_seq`. This will generate
        # the survival sequence for the case where the patient dies at the interval that `time` falls into.
        event = tf.ones_like(time)
        surv_seq = tf.cast(self.surv_seq(time, event), tf.float32)  # (n_obs x n_intervals) = (n_obs x m+1)
        # Unnormalized score for each bit that represents a single interval in a survival sequence
        scores = tf.matmul(features, self.w) + self.b  # (n_obs x n_times) = (n_obs x m)
        # For all monotonically increasing survival sequences, calculate the total unnormalized score for the
        # non-zero bits. This gives the unnormalized scores associated with observing all the different survival
        # sequences. These are needed to calculate the normalization factor.
        cumulative_scores = tf.matmul(scores, self.ltri)  # (n_obs x n_intervals)
        # Shifting the values by a constant safeguards against stack overflow during exponentiation
        shift = tf.maximum(tf.reduce_max(cumulative_scores, axis=1, keepdims=True), 0)  # (n_obs x 1)
        cumulative_scores = tf.exp(cumulative_scores - shift)  # (n_obs x m+1)
        # Calculate the normalization factor that converts scores into valid probabilities
        log_normalization = tf.math.log(tf.reduce_sum(cumulative_scores, axis=1, keepdims=True)) + shift  # (n_obs x 1)
        # The score of the survival sequence where death does not occur in any of the modeled intervals, but rather
        # in the final interval (which is not explicitly modeled by the model parameters), is defined as zero,
        # so zero is appended to the end of the score vector
        scores = tf.concat([scores, tf.zeros(shape=(tf.shape(scores)[0], 1), dtype=tf.float32)], axis=1)
        # Exclude scores for survival sequences that are not compatible with the time of death
        log_score = tf.reduce_sum(tf.multiply(scores, surv_seq), axis=1, keepdims=True)
        # Calculate the probability of observing death in the interval that includes the specified time
        surv_prob = tf.exp(log_score - log_normalization, name="surv_prob")  # (n_obs x 1)

        return surv_prob

    def call(self, inputs, *args, **kwargs):
        """Forward pass calculates the probability of observing death in each interval given some input features"""
        survival_distribution = tf.vectorized_map(
            lambda t: self.survival_fn(inputs, tf.repeat(t, tf.shape(inputs)[0])),
            self.interval_lower
        )  # (n_obs x m+1)

        return survival_distribution
