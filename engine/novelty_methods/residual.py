# ######################################################################
# Author: Cameron M. Gattis
# Created: 2026-06-13
# Updated: New.
# Purpose: Baseline novelty scoring method using mean absolute
#          reconstruction residuals.
# ######################################################################

import numpy as np

from .base import BaseNovelty


class ResidualNovelty(BaseNovelty):
    # ##################################################################
    # Author: Cameron M. Gattis
    # Created: 2026-06-13
    # Updated: New.
    # Purpose: Calculates novelty scores using raw reconstruction
    #          residuals without normalization.
    # ##################################################################

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
            Computes novelty score as the mean absolute
            reconstruction residual.

            @params x : numpy array
                    reconstruction : numpy array

            @return novelty : float
        '''
        residual = np.abs(x - reconstruction)

        return np.mean(residual)


    def explain(self, x, reconstruction):
        '''
            Returns feature-wise residual contributions.

            @params x : numpy array
                    reconstruction : numpy array

            @return residual : numpy array
        '''
        return np.abs(x - reconstruction)