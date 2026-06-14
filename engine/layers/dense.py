# ######################################################################
# Author: Cameron M. Gattis
# Created: 2026-06-13
# Updated: New.
# Purpose: Fully-connected neural network layer implementing
#          forward and backward propagation.
# ######################################################################

import numpy as np

from engine.layers.base_layer import Layer


class Dense(Layer):
    # ##################################################################
    # Author: Cameron M. Gattis
    # Created: 2026-06-13
    # Updated: New.
    # Purpose: Applies a linear transformation to input data
    #          using trainable weights and biases.
    # ##################################################################

    def __init__(self, inputs, outputs, seed=42):
        '''
            Initializes dense layer.

            @params inputs : int
                Number of input features.

                    outputs : int
                Number of output units.

                    seed : int
                Random initialization seed.
        '''
        rng = np.random.default_rng(seed)
        self.inputs = inputs
        self.outputs = outputs
        self.W = rng.normal(
            loc=0.0,
            scale=0.01,
            size=(inputs, outputs)
        )
        self.b = np.zeros(outputs)
        self.X = None

    def forward(self, X) -> np.ndarray:
        '''
            Performs forward propagation.

            Formula:

                Y = XW + b

            @params X : numpy array

            @return output : numpy array
        '''
        X = np.asarray(X, dtype=float)
        self.X = X

        return X @ self.W + self.b

    # layers/dense.py

    def backward(self,grad,learning_rate=0.001)  -> np.ndarray:
        '''
            Performs backpropagation and updates
            trainable parameters.

            @params grad : numpy array
                Gradient from subsequent layer.

                    learning_rate : float
                Optimization step size.

            @return grad_input : numpy array
                Gradient propagated to previous layer.
        '''
        grad_W = self.X.T @ grad
        grad_b = grad.sum(axis=0)
        self.W -= (learning_rate * grad_W)
        self.b -= (learning_rate * grad_b)
        grad_input = (grad @ self.W.T)

        return grad_input