import numpy as np

from .base import BaseActivation


class StatisticalActivation(
        BaseActivation
):

    def __init__(
            self,
            eps=1e-8
    ):

        self.eps = eps


    def forward(
            self,
            x
    ):

        mean = np.mean(
            x,
            axis=0
        )

        std = np.std(
            x,
            axis=0
        )

        z = (
            x -
            mean
        ) / (
            std +
            self.eps
        )

        return z


    def backward(
            self,
            x,
            grad
    ):

        return grad