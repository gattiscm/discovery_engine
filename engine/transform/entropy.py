# ######################################################################
# Author: Cameron M. Gattis
# Created: 2026-06-14
# Updated: New.
# Purpose: Entropy-based feature transform.
# ######################################################################

import numpy as np

from .base import BaseTransform


class EntropyTransform(BaseTransform):
    class EntropyTransform(BaseTransform):

    # ##################################################################
    # Author: Cameron M. Gattis
    # Created: 2026-06-14
    # Updated: New.
    # Purpose: Converts feature values into entropy-style
    #          contribution representations.
    # ##################################################################

    def __init__(self,eps=1e-8):
        '''
            Initializes entropy transform.

            @params eps : float
                Numerical stability constant.
        '''
        self.eps = eps


    def fit(self,X):
        '''
            No fitting required.

            Included for interface consistency.

            @params X : numpy array

            @return self : EntropyTransform
        '''
        return self


    def forward(self,X):
        '''
            Computes feature-wise entropy
            contributions.

            @params X : numpy array

            @return transformed : numpy array
        '''
        X_abs = np.abs(X)
        p = (X_abs / (np.sum(X_abs,axis=1,keepdims=True) + self.eps))

        return (-p * np.log(p + self.eps))