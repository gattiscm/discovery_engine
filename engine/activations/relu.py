import numpy as np

from .base import BaseActivation


class ReLU(
        BaseActivation
):

    def forward(
            self,
            x
    ):

        return np.maximum(
            0,
            x
        )


    def backward(
            self,
            x,
            grad
    ):

        return (
            grad *
            (x > 0)
        )