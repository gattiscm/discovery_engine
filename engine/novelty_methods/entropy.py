# ######################################################################
# Author: Cameron M. Gattis
# Created: 2026-06-13
# Updated: New.
# Purpose: Novelty scoring method based on entropy of
#          reconstruction residual distributions.
# ######################################################################

import numpy as np

from .base import BaseNovelty


class EntropyNovelty(BaseNovelty):
    # ##################################################################
    # Author: Cameron M. Gattis
    # Created: 2026-06-13
    # Updated: New.
    # Purpose: Calculates novelty scores using Shannon entropy
    #          of normalized reconstruction residuals.
    # ##################################################################

    def __init__(self, eps=1e-8):
        '''
            Initializes entropy novelty scorer.

            @params eps : float
                Small value used to prevent division by
                zero and log(0).
        '''
        self.eps = eps


    def fit(self, X):
        '''
            No fitting required.

            Included for interface compatibility with other
            novelty scoring methods.

            @params X : numpy array
        '''
        pass


    def compute(self, x, reconstruction):
        '''
            Computes entropy of normalized reconstruction
            residuals.

            Higher values indicate reconstruction errors
            distributed across many features.

            @params x : numpy array
                    reconstruction : numpy array

            @return entropy : float
        '''
        residual = np.abs(x - reconstruction)

        p = (residual / (np.sum(residual) + self.eps))

        entropy = -np.sum(p * np.log(p + self.eps))

        return entropy


    def explain(self, x, reconstruction):
        '''
            Returns normalized residual contribution
            probabilities.

            @params x : numpy array
                    reconstruction : numpy array

            @return p : numpy array
        '''
        residual = np.abs(x - reconstruction)

        p = (residual / (np.sum(residual) + self.eps))

        return p