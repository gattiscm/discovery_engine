import numpy as np
from scipy.special import expit
from engine.utils.architecture import (
    auto_latent_units
)

class StatisticalLayer:

    def __init__(
            self,
            units="auto",
            learning_rate=0.01,
            groups=None,
            mode="parallel",
            merge="concat"
    ):

        self.units = units
        self.learning_rate = learning_rate

        self.groups = groups
        self.mode = mode
        self.merge = merge

        self.weights = None
        self.bias = None

        self.X = None
        self.Z = None


    # ==========================================
    # initialize
    # ==========================================

    def initialize(
            self,
            input_dim
    ):

        if self.units == "auto":
            self.units = (
                auto_latent_units(
                    input_dim
                )
            )

            print(
                f"[StatisticalLayer] "
                f"Auto latent units: "
                f"{self.units}"
            )

        limit = np.sqrt(
            6 / (
                    input_dim +
                    self.units
            )
        )

        self.weights = np.random.uniform(
            -limit,
            limit,
            (
                input_dim,
                self.units
            )
        )

        self.bias = np.zeros(
            (
                1,
                self.units
            )
        )

        print(
            f"[StatisticalLayer] "
            f"Weights: {self.weights.shape}"
        )


    # ==========================================
    # activation
    # ==========================================

    def activation(self,x):

        return expit(x)


    def activation_derivative(
            self,
            x
    ):

        return x*(1-x)


    # ==========================================
    # forward
    # ==========================================

    def forward(
            self,
            X
    ):

        if self.weights is None:

            self.initialize(
                X.shape[1]
            )

        self.X = X

        linear = (

                X
                @
                self.weights

                +

                self.bias
        )

        self.Z = self.activation(
            linear
        )

        return self.Z


    # ==========================================
    # backward
    # ==========================================

    def backward(
            self,
            grad
    ):

        grad_local = (

                grad
                *
                self.activation_derivative(
                    self.Z
                )
        )

        dW = (

                self.X.T
                @
                grad_local
        )

        dB = np.sum(
            grad_local,
            axis=0,
            keepdims=True
        )

        grad_input = (

                grad_local
                @
                self.weights.T
        )

        self.weights -= (
                self.learning_rate
                *
                dW
        )

        self.bias -= (
                self.learning_rate
                *
                dB
        )

        return grad_input