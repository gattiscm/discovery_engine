import numpy as np

from .base import BaseActivation


class GroupedActivation(
        BaseActivation
):

    def __init__(
            self,
            groups,
            activation
    ):

        self.groups = groups
        self.activation = activation


    def forward(
            self,
            x
    ):

        output = np.zeros_like(
            x
        )

        for group in self.groups:

            output[:, group] = (
                self.activation.forward(
                    x[:, group]
                )
            )

        return output


    def backward(
            self,
            x,
            grad
    ):

        output = np.zeros_like(
            grad
        )

        for group in self.groups:

            output[:, group] = (
                self.activation.backward(
                    x[:, group],
                    grad[:, group]
                )
            )

        return output