# ######################################################################
# Author: Cameron M. Gattis
# Created: 2026-06-13
# Updated: New.
# Purpose: Statistical latent layer implementing trainable
#          feature projection with sigmoid activation.
# ######################################################################

import numpy as np
from scipy.special import expit
from engine.utils.architecture import auto_latent_units
from engine.activations import get_activation

class StatisticalLayer:
    # ##################################################################
    # Author: Cameron M. Gattis
    # Created: 2026-06-13
    # Updated: New.
    # Purpose: Learns latent statistical representations using
    #          a trainable linear transformation followed by a
    #          sigmoid activation function.
    # ##################################################################

    def __init__(
            self,
            units="auto",
            activation="sigmoid",
            learning_rate=0.01,
            groups=None,
            mode="parallel",
            merge="concat"
    ):
        '''
            Initializes statistical layer.

            @params units : int | str
                Number of latent units or "auto".

                    learning_rate : float
                Gradient descent learning rate.

                    groups : dict
                Optional feature grouping.

                    mode : str
                Group processing mode.

                    merge : str
                Group output merge strategy.
        '''

        self.units = units
        self.learning_rate = learning_rate

        self.groups = groups
        self.mode = mode
        self.merge = merge

        self.weights = None
        self.bias = None
        self.activation_name = activation

        (
            self.activation,
            self.activation_derivative
        ) = get_activation(
            activation
        )
        self.X = None
        self.Z = None


    # ==========================================
    # initialize
    # ==========================================

    def initialize(self,input_dim):
        '''
            Initializes weights and biases.

            Uses Xavier/Glorot uniform initialization.

            @params input_dim : int
                Number of input features.
        '''

        if self.units == "auto":
            self.units = (auto_latent_units(input_dim))
            print(f"[StatisticalLayer] Auto latent units: {self.units}")

        limit = np.sqrt(6 / (input_dim + self.units))

        self.weights = np.random.uniform(-limit,limit,(input_dim,self.units))
        self.bias = np.zeros((1,self.units))
        print(f"[StatisticalLayer] Weights: {self.weights.shape}")


    # ==========================================
    # forward
    # ==========================================

    def forward(self, X):
        '''
            Performs forward propagation.

            @params X : numpy array

            @return latent : numpy array
        '''
        if self.weights is None:
            self.initialize(X.shape[1])
        self.X = X
        linear = (X @ self.weights + self.bias)
        X_transform = self.transform.forward(X)

        linear = (
                X_transform
                @
                self.weights
                +
                self.bias
        )

        self.Z = self.activation.forward(
            linear
        )

        return self.Z


    # ==========================================
    # backward
    # ==========================================

    def backward(self, grad):
        '''
            Performs backpropagation and updates
            trainable parameters.

            @params grad : numpy array
                Gradient from subsequent layer.

            @return grad_input : numpy array
        '''
        grad_local = (grad * self.activation_derivative(self.Z))
        dW = (self.X.T @ grad_local)
        dB = np.sum(grad_local,axis=0,keepdims=True)
        grad_input = (grad_local @ self.weights.T)
        self.weights -= (self.learning_rate * dW)
        self.bias -= (self.learning_rate * dB)

        return grad_input