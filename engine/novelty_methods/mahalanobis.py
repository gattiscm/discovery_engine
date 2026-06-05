import numpy as np

from .base import BaseNovelty


class MahalanobisNovelty(
        BaseNovelty
):

    def __init__(
            self,
            regularization=1e-6
    ):

        self.regularization = regularization

        self.mean = None
        self.inv_cov = None


    def fit(
            self,
            X
    ):

        self.mean = np.mean(
            X,
            axis=0
        )

        cov = np.cov(
            X,
            rowvar=False
        )

        cov += (
            np.eye(
                cov.shape[0]
            ) *
            self.regularization
        )

        self.inv_cov = np.linalg.inv(
            cov
        )


    def compute(
            self,
            x,
            reconstruction
    ):

        residual = np.abs(
            x -
            reconstruction
        )

        d = np.sqrt(
            residual.T @
            self.inv_cov @
            residual
        )

        return d


    def explain(
            self,
            x,
            reconstruction
    ):

        residual = np.abs(
            x -
            reconstruction
        )

        return residual