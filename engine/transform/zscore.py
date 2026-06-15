# ######################################################################
# Author: Cameron M. Gattis
# Created: 2026-06-14
# Updated: Refactored.
# Purpose: Feature-wise z-score normalization transform.
# ######################################################################

import numpy as np

from .base import BaseTransform


class ZScoreTransform(BaseTransform):
    # ##################################################################
    # Author: Cameron M. Gattis
    # Created: 2026-06-14
    # Updated: Refactored.
    # Purpose: Normalizes features using configurable
    #          centering and scaling statistics.
    # ##################################################################

    def __init__(
            self,
            center="mean",
            scale="std",
            eps=1e-8
    ):
        '''
            Initializes z-score transform.

            @params center : str
                Centering statistic.

            @params scale : str
                Scaling statistic.

            @params eps : float
                Numerical stability constant.
        '''

        self.center_method = center
        self.scale_method = scale
        self.eps = eps

        self.center = None
        self.scale = None


    def fit(self,X):
        '''
            Learns normalization statistics.

            @params X : numpy array

            @return self : ZScoreTransform
        '''

        if self.center_method == "mean":
            self.center = np.mean(X,axis=0)

        elif self.center_method == "median":
            self.center = np.median(X,axis=0)

        else:
            raise ValueError(f"Unknown center method: {self.center_method}")

        if self.scale_method == "std":
            self.scale = np.std(X,axis=0)

        elif self.scale_method == "iqr":
            q75 = np.percentile(X,75,axis=0)
            q25 = np.percentile(X,25,axis=0)
            self.scale = (q75 - q25)

        else:
            raise ValueError(f"Unknown scale method: "{self.scale_method}")

        return self


    def forward(self, X):
        '''
            Applies feature normalization.

            @params X : numpy array

            @return transformed : numpy array

            @exception ValueError :
                Raised if fit() has not been called.
        '''

        if self.center is None:

            raise ValueError("ZScoreTransform must be fit before forward().")

        return (X - self.center) / (self.scale + self.eps)