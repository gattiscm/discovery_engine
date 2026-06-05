import numpy as np

from .base import BaseActivation


class Tanh(
        BaseActivation
):

    def forward(
            self,
            x
    ):

        return np.tanh(
            x
        )


    def backward(
            self,
            x,
            grad
    ):

        t = np.tanh(
            x
        )

        return (
            grad *
            (
                1 -
                t**2
            )
        )