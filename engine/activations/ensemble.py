import numpy as np

from .base import BaseActivation


class EnsembleActivation(
        BaseActivation
):

    def __init__(
            self,
            methods
    ):

        self.methods = methods


    def forward(
            self,
            x
    ):

        output = 0

        for method, weight in self.methods:

            output += (

                weight *
                method.forward(
                    x
                )

            )

        return output


    def backward(
            self,
            x,
            grad
    ):

        total = 0

        for method, weight in self.methods:

            total += (

                weight *
                method.backward(
                    x,
                    grad
                )

            )

        return total