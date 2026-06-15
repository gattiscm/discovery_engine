# ######################################################################
# Author: Cameron M. Gattis
# Created: 2026-06-09
# Updated: New.
# Purpose: Utility functions for novelty score calculation
#          and residual analysis.
# ######################################################################

import numpy as np


def compute_novelty(x,reconstruction):
    '''
        Computes weighted novelty score using
        reconstruction residuals.

        @params x : numpy array

                reconstruction : numpy array

        @return novelty : float
    '''
    residual=(x - reconstruction)
    weighted=(np.abs(residual) * (1 + np.abs(x)))

    return float(np.mean(weighted))


def compute_residual_vector(x, reconstruction):
    '''
        Computes feature-wise reconstruction
        residuals.

        @params x : numpy array

                reconstruction : numpy array

        @return residual : numpy array
    '''
    x = np.asarray(x, dtype=float)
    reconstruction = np.asarray(reconstruction, dtype=float)

    return np.abs(x - reconstruction)