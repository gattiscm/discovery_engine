# ######################################################################
# Author: Cameron M. Gattis
# Created: 2026-06-13
# Updated: New.
# Purpose: Statistical feature transform layer supporting
#          configurable feature-space transformations.
# ######################################################################

import numpy as np

from .base import BaseTransform


class StatisticalTransform(BaseTransform):
    # ##################################################################
    # Author: Cameron M. Gattis
    # Created: 2026-06-13
    # Updated: New.
    # Purpose: Applies configurable statistical feature
    #          transformations prior to latent projection.
    # ##################################################################

    def __init__(
            self,
            method="zscore",
            eps=1e-8
    ):
        '''
            Initializes statistical transform.

            @params method : str
                Statistical transformation type.

                    eps : float
                Numerical stability constant.
        '''

        self.method = method
        self.eps = eps

        self.mean = None
        self.std = None


    def fit(
            self,
            X
    ):
        '''
            Learns required statistics from data.

            @params X : numpy array
        '''

        if self.method == "zscore":

            self.mean = np.mean(
                X,
                axis=0
            )

            self.std = np.std(
                X,
                axis=0
            )


    def forward(
            self,
            X
    ):
        '''
            Applies statistical transformation.

            @params X : numpy array

            @return transformed : numpy array
        '''

        if self.method == "identity":

            return X

        if self.method == "zscore":

            return (
                X - self.mean
            ) / (
                self.std + self.eps
            )

        raise ValueError(
            f"Unknown transform: "
            f"{self.method}"
        )