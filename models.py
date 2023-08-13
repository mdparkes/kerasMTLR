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
from typing import Any, Callable, List, Optional, Tuple, Union

from layers import MTLR


ArrayLike = Union[List, np.ndarray, Tensor]


class NeuralNetworkMTLRSurvival(tf.keras.Model):
    """
    An end-to-end model for predicting survival using a multi-task logistic regression (MTLR). The user can optionally
    supply a keras.Layer argument to the nn_block __init__ parameter to calculate a latent representation for input
    to MTLR. The neural network block will be optimized using the MTLR negative log likelihood of survival data as
    the loss minimization objective. If nn_block is not provided by the user, the features of the input data records
    will be used directly for MTLR.

    This model is only compatible with right-censored data.

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

    The MTLR layer's forward pass returns an array of probabilities that a patient's time of death was greater than or
    equal to the lower bound of each of the intervals that were used to partition time. The probability of an event
    occurring in any particular interval can be calculated by a forward pass of the NeuralNetworkMTLRSurvival.mtlr
    Layer.
    """
    def __init__(self,
                 nn_block: Optional[tf.keras.layers.Layer],
                 kernel_initializer: Union[str, Callable] = "random_normal",
                 bias_initializer: Union[str, Callable] = "random_normal",
                 kernel_regularizer: Union[None, Callable] = None,
                 bias_regularizer: Union[None, Callable] = None,
                 *,
                 interval_lower: Optional[ArrayLike] = None,
                 interval_upper: Optional[ArrayLike] = None,
                 **kwargs):
        """
        :param nn_block:  An optional Layer that calculates a latent representation of the data for MTLR.
        :param kernel_initializer: The weight initialization to use
        :param bias_initializer: The bias initialization to use
        :param interval_lower: An array of lower bounds on time intervals for MTLR modeling. Only one of interval_lower
        or interval_upper needs to be specified.
        :param interval_upper: An array of upper bounds on time intervals for MTLR modeling.
        :param kwargs: Other keyword arguments used by the keras.Model __init__ method
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

        kwargs.setdefault("name", "nn_mtlr_model")
        super().__init__(**kwargs)
        self.nn_block = nn_block
        self.mtlr = MTLR(kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                         kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                         interval_lower=interval_lower, interval_upper=interval_upper)  # MTLR layer
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")  # Running average of training example losses

    def get_config(self):
        config = super(NeuralNetworkMTLRSurvival, self).get_config()
        interval_lower = self.mtlr.interval_lower.numpy().tolist()  # Makes it JSON-compatible for serialization
        interval_upper = self.mtlr.interval_upper.numpy().tolist()
        config.update(
            {"nn_block": self.nn_block, "kernel_initializer": self.mtlr.kernel_initializer,
             "bias_initializer": self.mtlr.bias_initializer, "kernel_regularizer": self.mtlr.kernel_regularizer,
             "bias_regularizer": self.mtlr.bias_regularizer, "interval_lower": interval_lower,
             "interval_upper": interval_upper}
        )
        return config

    @classmethod
    def from_config(cls, config, **kwargs):
        return cls(**config)

    @property
    def metrics(self):
        return [self.loss_tracker]

    @tf.custom_gradient
    def loss_fn(self, features: Tensor, time: Tensor, event: Tensor) -> Tuple[Tensor, Callable[..., Any]]:
        """Calculate the negative log likelihood of an observed death or censoring time given the features. Custom
        gradients are calculated with respect to the MTLR layer's trainable parameters and inputs because they cannot
        be autodifferentiated, but autodifferentation is used to learn the trainable parameters of the ANN block that
        gives the input to the MTLR layer (unless the block specifies its own custom gradients). Different gradients
        and losses are calculated for an input to the MTLR layer depending on whether the observation was censored or
        uncensored at the indicated time. Left-censored events should be indicated by -1, uncensored by 1,
        and right-censored by 0."""

        time = tf.reshape(time, (-1,))  # Reshape as rank 1 tensor, i.e., vector
        # Cast event as int32 so it can be used with tf.dynamic_partition. A value of -1 indicates left censored
        # data, 0 indicates right-censored data, and 1 indicates uncensored data.
        # Dynamic partitioning can only use non-negative integers, so left-censoring values of -1 are substituted with 2
        event = tf.where(event == -1, tf.fill(tf.shape(event), 2), event)
        event = tf.cast(tf.reshape(event, (-1,)), tf.int32)
        n_obs = tf.shape(features)[0]

        def true_fn():  # For tf.case() below, executed if a condition is satisfied
            return tf.reshape(features, shape=(1, -1))  # (n_obs x p)

        # If the MTLR input is rank 1 (a vector), reshape it to a rank 2 row vector (a matrix with one row)
        features = tf.case([(tf.equal(tf.rank(features), 1), true_fn)], default=lambda: features)
        surv_seq = self.mtlr.surv_seq(time, event)  # Binary survival sequences of training examples in the batch
        rc_seq, uc_seq, lc_seq = tf.dynamic_partition(surv_seq, event, 3)

        # Below several objects are created to calculate the losses and gradients.
        # linear_scores is a (n_obs x m) matrix of real-valued, untransformed scores for observing a 1 at each of m
        #   intervals in the binary survival sequences for n_obs event/censoring times. There are m+1 intervals in
        #   the survival sequence, but only the parameters for first m intervals are learned.
        # cum_scores is an (n_obs x m+1) matrix of exponentiated sums of linear scores for each possible survival
        #   sequence from [1, 1, ..., 1, 1] to [0, 0, ..., 0, 1] for each of n_obs examples. In other words,
        #   for each example it is a vector of scores associated with every possible survival sequence outcome. Each
        #   score can be divided by a normalization factor to give the probability of observing a particular survival
        #   sequence given the patient's features and the current model parameters.
        # log_normalization is the normaliztion factor that transforms unnormalized cum_scores into probabilities
        # uncns_log_score is the unnormalized score associated with observing the survival sequence for the observed
        #   event time.
        # lc_log_score is the unnormalized sum of scores associated with each survival sequence that is possible
        #   given the left censoring time.
        # rc_log_score is the unnormalized sum of scores associated with each survival sequence that is possible
        #   given the right censoring time.
        # shift is a factor that is used to make exponentiation numerically stable.
        # self.mtlr.ltri is an (m x m+1) augmented lower triangular binary matrix whose columns represent each possible
        #   survival sequence that can be modeled by MTLR. It is used for various linear algebra operations in the
        #   loss and gradient calculations. Its final column is a zero vector.

        # Calculate batch-wide scores and normalization factors
        linear_scores = tf.matmul(features, self.mtlr.w) + self.mtlr.b  # (n_obs x m)
        cum_scores = tf.matmul(linear_scores, self.mtlr.ltri)  # (n_obs x m+1)
        shift = tf.maximum(tf.reduce_max(cum_scores, axis=1, keepdims=True), 0)  # (n_obs x 1)
        cum_scores = tf.exp(cum_scores - shift)  # vector function f(x, a) -> (n_obs x m+1)
        log_normalization = tf.math.log(tf.reduce_sum(cum_scores, axis=1, keepdims=True)) + shift  # (n_obs x 1)

        # Partition the set of input example tensors into three partitions according to censor status
        obs_idx = tf.range(tf.shape(features)[0])  # Indices of each example in the data set
        condition_indices = tf.dynamic_partition(obs_idx, event, 3)  # For restitching
        # lc: left censored (event = -1); rc: right censored (event = 0); uc: uncensored (event = 1)
        rc_input, uc_input, lc_input = tf.dynamic_partition(features, event, 3)  # List of partitions
        rc_cum_scores, uc_cum_scores, lc_cum_scores = tf.dynamic_partition(cum_scores, event, 3)  # (n_cns|n_uncns x 1)
        rc_shift, uc_shift, lc_shift = tf.dynamic_partition(shift, event, 3)  # (n_cns|n_uncns x 1)
        rc_log_norm, uc_log_norm, lc_log_norm = tf.dynamic_partition(log_normalization, event, 3)  # (n_cns|n_uncns x 1)

        # Calculate left-censored losses for each example in the batch
        lc_log_score = tf.math.log(tf.reduce_sum(tf.multiply(lc_cum_scores, lc_seq), axis=1, keepdims=True))
        # The censored log score is the log of an exponentiated sum of terms. To make the exponentiation numerically
        # stable, we shifted each term by subtracting a value. The shift is negated by adding the value back to
        # lc_log_score below.
        lc_log_score = lc_log_score + lc_shift
        lc_loss = -1 * (lc_log_score - lc_log_norm)  # (n_cns x 1), neg. log. likelihood

        # Calculate right-censored losses for each example in the batch
        rc_log_score = tf.math.log(tf.reduce_sum(tf.multiply(rc_cum_scores, rc_seq), axis=1, keepdims=True))
        # The censored log score is the log of an exponentiated sum of terms. To make the exponentiation numerically
        # stable, we shifted each term by subtracting a value. The shift is negated by adding the value back to
        # rc_log_score below.
        rc_log_score = rc_log_score + rc_shift
        rc_loss = -1 * (rc_log_score - rc_log_norm)  # (n_cns x 1), neg. log. likelihood

        # Calculate uncensored losses for each example in the batch
        _, uncns_scores, _ = tf.dynamic_partition(linear_scores, event, 3)
        n_uncns = tf.shape(uncns_scores)[0]
        uncns_scores = tf.concat([uncns_scores, tf.zeros(shape=(n_uncns, 1), dtype=tf.float32)], axis=1)
        # The "real" uncensored log score is the log of a single exponentiated term, but taking the log nullifies
        # the exponentiation, so we omit both operations below in an equivalent expression. This removes the need to
        # adjust the log score by the shift, since shift is only used to guarantee numeric stability when the
        # exponentiation is actually calculated, as it was for the censored score.
        uncns_log_score = tf.reduce_sum(tf.multiply(uncns_scores, uc_seq), axis=1, keepdims=True)
        uncns_loss = -1 * (uncns_log_score - uc_log_norm)  # (n_uncns x 1), neg. log. likelihood

        mtlr_loss_result = tf.dynamic_stitch(condition_indices, [rc_loss, uncns_loss, lc_loss])  # (n_obs x 1)
        mtlr_loss_result = tf.reduce_mean(mtlr_loss_result)  # Mean negative log likelihood for the batch

        def gradient(grad, variables):
            """grad is upstream gradients used in chain rule to calculate gradients; variables is trainable
            parameters accessed by get_variables methods and used in the scope of mtlr_loss calculations"""
            # All gradients are multiplied by a scaling factor so that the censored and uncensored gradients sum to
            # the combined per-sample average over the entire batch
            assert variables is not None
            assert len(variables) == 2
            assert variables[0] is self.mtlr.b and variables[1] is self.mtlr.w
            # NB: if a trainable weights' name attribute is set when they are created with the add_weight
            # method (using the name kwarg), they will be passed to the variables parameter in alphabetical
            # order. If the name is unspecified, it appears that they will be passed to variables in the order they
            # were created, although I have not thoroughly tested this assumption.

            variables_grad = []

            # region Left-censored partial gradients
            valid_scores = tf.multiply(lc_cum_scores, lc_seq)
            # Since lc_cum_scores was calculated using the shift, we need to use a log_norm term that is calculated
            # from shifted scores to get norm1. This is achieved by taking the unshifted log_norm and subtracting the
            # shift value.
            norm1 = tf.exp(lc_log_norm - lc_shift)  # Normalization of all survival sequences
            norm2 = tf.reduce_sum(valid_scores, axis=1, keepdims=True)
            reusable_term = (lc_cum_scores / norm1) - (valid_scores / norm2)  # (n_cns x m+1)
            reusable_term = tf.matmul(self.mtlr.ltri, reusable_term, transpose_b=True)  # (m x n_cns)
            lc_part_feat_grad = tf.transpose(tf.matmul(self.mtlr.w, reusable_term))  # (n_cns x p)
            lc_part_w_grad = tf.vectorized_map(
                lambda a: tf.tensordot(a[0], a[1], axes=0),  # Outer product -> (n_cns x p x m)
                [lc_input, tf.transpose(reusable_term)]  # (n_cns x p), (n_cns x m)
            )  # (n_cns x p x m)
            lc_part_b_grad = tf.transpose(reusable_term)  # (n_cns x m)
            # endregion Left-censored partial gradients

            # region Right-censored partial gradients
            valid_scores = tf.multiply(rc_cum_scores, rc_seq)
            # Since rc_cum_scores was calculated using the shift, we need to use a log_norm term that is calculated
            # from shifted scores to get norm1. This is achieved by taking the unshifted log_norm and subtracting the
            # shift value.
            norm1 = tf.exp(rc_log_norm - rc_shift)  # Normalization of all survival sequences
            norm2 = tf.reduce_sum(valid_scores, axis=1, keepdims=True)
            reusable_term = (rc_cum_scores / norm1) - (valid_scores / norm2)  # (n_cns x m+1)
            reusable_term = tf.matmul(self.mtlr.ltri, reusable_term, transpose_b=True)  # (m x n_cns)
            rc_part_feat_grad = tf.transpose(tf.matmul(self.mtlr.w, reusable_term))  # (n_cns x p)
            rc_part_w_grad = tf.vectorized_map(
                lambda a: tf.tensordot(a[0], a[1], axes=0),  # Outer product -> (n_cns x p x m)
                [rc_input, tf.transpose(reusable_term)]  # (n_cns x p), (n_cns x m)
            )  # (n_cns x p x m)
            rc_part_b_grad = tf.transpose(reusable_term)  # (n_cns x m)
            # endregion Right-censored partial gradients

            # region Uncensored partial gradients
            seq_trunc = uc_seq[:, :-1]  # Drop the final interval's bit from the survival sequence
            norm = tf.exp(uc_log_norm - uc_shift)  # Normalization of all survival sequences
            reusable_term = tf.matmul(uc_cum_scores, self.mtlr.ltri, transpose_b=True) / norm
            reusable_term = reusable_term - seq_trunc  # (n_uncns x m)
            uc_part_feat_grad = tf.transpose(tf.matmul(self.mtlr.w, reusable_term, transpose_b=True))
            uc_part_w_grad = tf.vectorized_map(
                lambda a: tf.tensordot(a[0], a[1], axes=0),  # Outer product -> (n_uncns x p x m)
                [uc_input, reusable_term]  # (n_uncns x p), (n_uncns x m)
            )  # (n_uncns x p x m)
            uc_part_b_grad = reusable_term  # (n_uncns x m)
            # endregion Uncensored partial gradients

            partial_feat_grad = tf.dynamic_stitch(condition_indices,
                                                  [rc_part_feat_grad, uc_part_feat_grad, lc_part_feat_grad])
            partial_w_grad = tf.dynamic_stitch(condition_indices,
                                               [rc_part_w_grad, uc_part_w_grad, lc_part_w_grad])
            partial_b_grad = tf.dynamic_stitch(condition_indices,
                                               [rc_part_b_grad, uc_part_b_grad, lc_part_b_grad])

            features_grad = grad * partial_feat_grad / tf.cast(n_obs, tf.float32)
            time_grad = None
            event_grad = None
            w_grad = grad * tf.reduce_mean(partial_w_grad, axis=0)
            b_grad = grad * tf.reduce_mean(partial_b_grad, axis=0)

            variables_grad.append(b_grad)
            variables_grad.append(w_grad)

            return (features_grad, time_grad, event_grad), variables_grad

        return mtlr_loss_result, gradient

    @tf.function
    def train_step(self, data):
        """Custom training step executed by Model.fit()"""
        features, time, event = data
        if not self.mtlr.trainable_weights:  # Initializes weights
            _ = self(features)
        with tf.GradientTape() as tape:
            features = self.nn_block(features)
            loss = self.loss_fn(features, time, event)
        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    @tf.function
    def test_step(self, data):
        """Custom evaluation step executed by Model.evaluate()"""
        features, time, event = data
        features = self.nn_block(features)
        loss = self.loss_fn(features, time, event)
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def call(self, inputs, *args, **kwargs):
        if self.nn_block is not None:
            latent_representation = self.nn_block(inputs)
            survival_distribution = self.mtlr(latent_representation)
        else:
            survival_distribution = self.mtlr(inputs)
        # Reshape: (n_obs x n_intervals)
        survival_distribution = tf.transpose(tf.reshape(survival_distribution, [self.mtlr.n_intervals, -1]))
        # Calculate the cumulative distribution specifying the probability of surviving at least until a given interval
        cumulative_distribution = tf.vectorized_map(
            lambda x: tf.reverse(tf.cumsum(tf.reverse(x, axis=[-1])),  axis=[-1]),
            survival_distribution
        )
        return cumulative_distribution
