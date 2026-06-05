import numpy as np

from .base import BaseNovelty


class EntropyNovelty(
        BaseNovelty
):

    def __init__(
            self,
            eps=1e-8
    ):

        self.eps = eps


    def fit(
            self,
            X
    ):

        pass


    def compute(
            self,
            x,
            reconstruction
    ):

        residual = np.abs(
            x -
            reconstruction
        )

        p = (
            residual /
            (
                np.sum(
                    residual
                ) +
                self.eps
            )
        )

        entropy = -np.sum(
            p *
            np.log(
                p +
                self.eps
            )
        )

        return entropy


    def explain(
            self,
            x,
            reconstruction
    ):

        residual = np.abs(
            x -
            reconstruction
        )

        p = (
            residual /
            (
                np.sum(
                    residual
                ) +
                self.eps
            )
        )

        return p