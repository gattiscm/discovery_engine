import numpy as np

from .base import BaseNovelty


class ResidualNovelty(
        BaseNovelty
):

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

        return np.mean(
            residual
        )

    def explain(
            self,
            x,
            reconstruction
    ):

        return np.abs(
            x -
            reconstruction
        )