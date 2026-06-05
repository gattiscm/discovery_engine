import numpy as np

from .base import BaseActivation


class Sigmoid(
        BaseActivation
):

    def forward(
            self,
            x
    ):

        return 1.0 / (
            1.0 +
            np.exp(-x)
        )


    def backward(
            self,
            x,
            grad
    ):

        s = self.forward(
            x
        )

        return (
            grad *
            s *
            (1 - s)
        )