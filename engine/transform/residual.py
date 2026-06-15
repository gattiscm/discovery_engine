# ######################################################################
# Author: Cameron M. Gattis
# Created: 2026-06-14
# Updated: New.
# Purpose: Residual feature transform.
# ######################################################################

import numpy as np

from .base import BaseTransform


class ResidualTransform(BaseTransform):
    # ##################################################################
    # Author: Cameron M. Gattis
    # Created: 2026-06-14
    # Updated: New.
    # Purpose: Converts features into absolute residuals
    #          relative to a reference statistic.
    # ##################################################################

    def __init__(
            self,
            reference="mean"
    ):
        '''
            Initializes residual transform.

            @params reference : str
                Reference statistic.

                    mean
                    median
        '''

        self.reference_method = reference
        self.reference = None


    def fit(self,X):
        '''
            Learns reference values.

            @params X : numpy array

            @return self : ResidualTransform
        '''

        if self.reference_method == "mean":
            self.reference = np.mean(X,axis=0)

        elif self.reference_method == "median":
            self.reference = np.median(X,axis=0)

        else:
            raise ValueError(f"Unknown reference method: {self.reference_method}")

        return self


    def forward(self,X):
        '''
            Computes absolute residuals.

            @params X : numpy array

            @return transformed : numpy array

            @exception ValueError :
                Raised if fit() has not been called.
        '''

        if self.reference is None:
            raise ValueError("ResidualTransform must be fit before forward().")

        return np.abs(X - self.reference)