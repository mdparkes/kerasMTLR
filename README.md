# kerasMTLR
An implementation of the MTLR algorithm for survival prediction using TensorFlow with Keras. Based on the paper ["Distributions as a Sequence of Dependent Regressors"](http://www.cs.cornell.edu/~cnyu/papers/nips11_survival.pdf) by Yu *et al* (NIPS 2011), but does not use the second parameter smoothing regularizer, as it has been shown that this regularizer is unnecessary.

## Details
The `NeuralNetworkMTLRSurvival` class extends `keras.Model` and allows the user to model individualized survival distributions from a latent representation of observed input features using MTLR. See below for a more detailed description of the MTLR algorithm. This model works with right-censored data, but does not support left- or interval-censored data. Its forward pass outputs a cumulative distribution of probabilities that a patient will survive at least until the beginning of each of a series of time intervals modeled by MTLR.

The `NeuralNetworkMTLRSurvival` `Model` can be initialized with a callable `keras.Layer` object which calculates a latent representation of the input examples' observable features. The latent representation is used as input to the final `MTLR` `Layer` of the model, and is learned in a supervised, end-to-end fashion using the loss minimization objective of the `MTLR` `Layer`. Specifically, the objective is to find model parameters that maximize the likelihood (minimize the negative log likelihood) of observing the training survival data.

Note that Keras' own `Sequential` class is not compatible with `NeuralNetworkMTLRSurvival` because of how the latter calculates the loss from its inputs. If the user wishes to use a sequential block of Keras `Layer`s to calculate a latent representation of their data, they can use the `SequentialBlock` `Layer` subclass in `layers.py`. `SequentialBlock.__init__` takes a list of Keras `Layer`s and constructs a sequential block with the `Layer`s stacked in the order they appear in the list.

Supplying a `keras.Layer` to the `NeuralNetworkMTLRSurvival` initializer is optional. If the user does not provide an upstream `Layer` that learns a latent representation for input to the MTLR output `Layer`, the MTLR output `Layer` will simply use the features of the data records as input. This is useful if the user wants to supply a previously learned unsupervised representation of their data as input to MTLR, or if they simply want to use the observed features.

## MTLR Details
Time (measured in units of the training data) is partitioned into m+1 right-open intervals that span 0 to infinity. For each patient we can construct a monotonically increasing survival sequence of m+1 bits which expresses all the intervals through which the patient survived (denoted by a value of 0) and all the intervals that the patient did not live through (denoted by a value of 1). Once a value of 1 is assigned, all bits thereafter must also be 1 (once dead, we tend to stay dead). For example, if we partition time into 5 intervals (i.e. m=4) and a patient's time of death was in the third interval, their survival sequence would be [0, 0, 1, 1, 1].

The trainable parameters of the MTLR layer maximize the likelihood of the observed times of death or censoring in the training dataset, given the associated input features. If a patient is censored (i.e. we only know the lower bound on their time of death, but not their true time of death), MTLR models the likelihood by marginalizing over all possible survival sequences that are compatible with the censoring time (the lower bound on time of death). If the patient is uncensored (the time of death is known), MTLR models the likelihood by considering the one survival sequence that represents the patient's known time of death.

The `MTLR` `Layer`'s forward pass returns an array of probabilities that a patient's time of death was greater than or equal to the lower bound of each of the intervals that were used to partition time. The probability of an event occurring in any particular interval can be calculated by a forward pass of the `NeuralNetworkMTLRSurvival.mtlr` `Layer`.
