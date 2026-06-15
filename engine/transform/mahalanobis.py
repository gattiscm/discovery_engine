# ######################################################################
# Author: Cameron M. Gattis
# Created: 2026-06-14
# Updated: New.
# Purpose: Mahalanobis-distance feature transform.
# ######################################################################

import numpy as np

from .base import BaseTransform


class MahalanobisTransform(BaseTransform):
    # ##################################################################
    # Author: Cameron M. Gattis
    # Created: 2026-06-14
    # Updated: New.
    # Purpose: Produces covariance-aware feature
    #          representations using Mahalanobis
    #          distance contributions.
    # ##################################################################

    def __init__(self,regularization=1e-6):
        '''
            Initializes Mahalanobis transform.

            @params regularization : float
                Diagonal covariance regularization.
        '''
        self.regularization = regularization
        self.mean = None
        self.inv_cov = None


    def fit(self,X):
        '''
            Learns feature covariance structure.

            @params X : numpy array

            @return self : MahalanobisTransform
        '''
        self.mean = np.mean(X,axis=0)
        cov = np.cov(X,rowvar=False)
        cov += (np.eye(cov.shape[0]) * self.regularization)
        self.inv_cov = (np.linalg.pinv(cov))

        return self


    def forward(self,X):
        '''
            Computes feature-level Mahalanobis
            contribution representation.

            @params X : numpy array

            @return transformed : numpy array

            @exception ValueError :
                Raised if fit() has not been called.
        '''
        if self.mean is None:
            raise ValueError("MahalanobisTransform must be fit before forward().")
        delta = (X - self.mean)

        return (delta * (delta @ self.inv_cov))