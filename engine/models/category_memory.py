# ######################################################################
# Author: Cameron M. Gattis
# Created: 2026-06-15
# Updated: New.
# Purpose: Stores statistical representations of known categories
#          for category-aware reconstruction and comparison.
# ######################################################################

import numpy as np


class CategoryMemory:
    # ##################################################################
    # Author: Cameron M. Gattis
    # Created: 2026-06-15
    # Updated: New.
    # Purpose: Maintains per-category statistical memory consisting
    #          of feature means and standard deviations.
    # ##################################################################

    def __init__(self):
        '''
            Initializes category memory storage.

            @params None

            @return None
        '''
        self.basis = {}

    def fit(self, X, y):
        '''
            Learns statistical representations for each category.

            @params X : numpy array
                    Feature matrix.

                    y : numpy array
                    Category labels.

            @return self : CategoryMemory
        '''
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        classes = np.unique(y)

        for c in classes:
            Xc = X[y == c]
            self.basis[c] = {"mean": np.mean(Xc, axis=0),
                "std": np.std(Xc, axis=0)}

        return self

    def reconstruct(self, category):
        '''
            Returns the stored mean feature vector for a category.

            @params category : object
                    Category label.

            @return mean : numpy array

            @exception ValueError : Raised when category
                                    is unknown.
        '''
        if category not in self.basis:
            raise ValueError(f"Unknown category: {category}")

        return self.basis[category]["mean"]

    def categories(self):
        '''
            Returns all known category labels.

            @params None

            @return categories : list
        '''
        return list(self.basis.keys())