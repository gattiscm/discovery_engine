# ######################################################################
# Author: Cameron M. Gattis
# Created: 2026-06-09
# Updated: New.
# Purpose: Novelty scoring method based on reconstruction residual
#          z-scores and root-mean-square aggregation.
# ######################################################################


import numpy as np

from .base import BaseNovelty


class ZScoreRMS(BaseNovelty):
    # ######################################################################
    # Author: Cameron M. Gattis
    # Created: 2026-06-09
    # Updated: New.
    # Purpose: alculates novelty scores using reconstruction
    #          residual z-scores.
    # ######################################################################
    def __init__(
            self,
            eps=1e-8):
        self.eps = eps
        self.mean = None
        self.std = None


    def fit(self,X):
        '''
            Learns feature-wise statistics from training data.

            @params X : numpy array
        '''
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X,axis=0)


    def _compute_z(self,x,reconstruction):
        '''
            Calculates z score from array.

            @params x : array
                    reconstruction : numpy array

            @return z : float

            @exception ValueError : Raised on uncalculated errors.
        '''
        if self.std is None:
            raise ValueError("ZScoreRMS must be fit before compute().")
        residual = np.abs(x - reconstruction)
        z = (residual / (self.std + self.eps))

        return z


    def compute(self,x,reconstruction):
        '''
            Computes RMS  novelty score.

            @params x : array
                    reconstruction : numpy object

            @return novelty : float
        '''
        z = self._compute_z(x,reconstruction)
        novelty = np.sqrt(np.mean(z ** 2))

        return novelty


    def explain(self,x,reconstruction):
        '''
            Returns feature-wise z-score contributions.

            @params x : array
                    reconstruction : numpy object

            @return calculated : float
        '''
        return self._compute_z(x,reconstruction)