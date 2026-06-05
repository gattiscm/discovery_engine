import numpy as np

from .base import BaseNovelty


class ZScoreRMS(
        BaseNovelty
):

    def __init__(
            self,
            eps=1e-8
    ):

        self.eps = eps

        self.mean = None
        self.std = None


    def fit(
            self,
            X
    ):

        self.mean = np.mean(
            X,
            axis=0
        )

        self.std = np.std(
            X,
            axis=0
        )


    def _compute_z(
            self,
            x,
            reconstruction
    ):

        if self.std is None:

            raise ValueError(
                "ZScoreRMS must be fit "
                "before compute()."
            )

        residual = np.abs(
            x -
            reconstruction
        )

        z = (
            residual /
            (
                self.std +
                self.eps
            )
        )

        return z


    def compute(
            self,
            x,
            reconstruction
    ):

        z = self._compute_z(
            x,
            reconstruction
        )

        novelty = np.sqrt(
            np.mean(
                z**2
            )
        )

        return novelty


    def explain(
            self,
            x,
            reconstruction
    ):

        return self._compute_z(
            x,
            reconstruction
        )