# ######################################################################
# Author: Cameron M. Gattis
# Created: 2026-06-13
# Updated: New.
# Purpose: Novelty scoring method using Mahalanobis distance on
#          reconstruction residuals.
# ######################################################################

import numpy as np

from .base import BaseNovelty


class MahalanobisNovelty(BaseNovelty):
    # ##################################################################
    # Author: Cameron M. Gattis
    # Created: 2026-06-13
    # Updated: New.
    # Purpose: Calculates novelty scores using covariance-aware
    #          Mahalanobis distance of reconstruction residuals.
    # ##################################################################

    def __init__(self, regularization=1e-6):
        '''
            Initializes Mahalanobis novelty scorer.

            @params regularization : float
                Small value added to covariance diagonal
                for numerical stability.
        '''
        self.regularization = regularization
        self.mean = None
        self.inv_cov = None


    def fit(self, X):
        '''
            Learns covariance structure from training data.

            @params X : numpy array
        '''
        self.mean = np.mean(X, axis=0)

        cov = np.cov(X, rowvar=False)
        cov += (np.eye(cov.shape[0]) * self.regularization)

        self.inv_cov = np.linalg.inv(cov)


    def compute(self, x, reconstruction):
        '''
            Computes Mahalanobis novelty score from
            reconstruction residuals.

            @params x : numpy array
                    reconstruction : numpy array

            @return novelty : float

            @exception ValueError : Raised if fit()
                                     has not been called.
        '''
        if self.inv_cov is None:
            raise ValueError(
                "MahalanobisNovelty must be fit before compute()."
            )

        residual = np.abs(x - reconstruction)

        d = np.sqrt(
            residual.T @ self.inv_cov @ residual
        )

        return d


    def explain(self, x, reconstruction):
        '''
            Returns feature-wise residual contributions.

            @params x : numpy array
                    reconstruction : numpy array

            @return residual : numpy array
        '''
        residual = np.abs(x - reconstruction)

        return residual