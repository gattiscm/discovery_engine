# ######################################################################
# Author: Cameron M. Gattis
# Created: 2026-06-13
# Updated: New.
# Purpose: Combines multiple novelty scoring methods into a
#          weighted ensemble score.
# ######################################################################

import numpy as np

from .base import BaseNovelty


class EnsembleNovelty(BaseNovelty):
    # ##################################################################
    # Author: Cameron M. Gattis
    # Created: 2026-06-13
    # Updated: New.
    # Purpose: Aggregates novelty scores from multiple methods
    #          using weighted summation.
    # ##################################################################

    def __init__(self, methods):
        '''
            Initializes ensemble novelty scorer.

            @params methods : list
                List of tuples in the form:

                (novelty_method, weight)
        '''
        self.methods = methods


    def fit(self, X):
        '''
            Fits all novelty methods in ensemble.

            @params X : numpy array
        '''
        for method, weight in self.methods:
            method.fit(X)


    def compute(self, x, reconstruction):
        '''
            Computes weighted ensemble novelty score.

            @params x : numpy array
                    reconstruction : numpy array

            @return novelty : float
        '''
        total = 0.0

        for method, weight in self.methods:
            score = method.compute(x,reconstruction)
            total += (weight * score)

        return total


    def explain(self, x, reconstruction):
        '''
            Returns contribution from each novelty
            scoring method.

            @params x : numpy array
                    reconstruction : numpy array

            @return explanation : dict
        '''
        explanation = {}

        for method, weight in self.methods:

            name = method.__class__.__name__

            explanation[name] = {
                "weight": weight,
                "score": method.compute(x,reconstruction)
            }

        return explanation