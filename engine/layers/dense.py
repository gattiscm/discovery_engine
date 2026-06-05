# layers/dense.py

import numpy as np

from engine.layers.base_layer import Layer


class Dense(Layer):

    def __init__(self, inputs, outputs, seed=42):
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

    def forward(self, X):
        X = np.asarray(X, dtype=float)

        self.X = X

        return X @ self.W + self.b

    # layers/dense.py

    def backward(
            self,
            grad,
            learning_rate=0.001
    ):
        grad_W = self.X.T @ grad
        grad_b = grad.sum(axis=0)

        self.W -= (
                learning_rate * grad_W
        )

        self.b -= (
                learning_rate * grad_b
        )

        grad_input = (
                grad @ self.W.T
        )

        return grad_input