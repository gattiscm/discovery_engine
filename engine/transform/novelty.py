# ######################################################################
# Author: Cameron M. Gattis
# Created: 2026-06-14
# Updated: New.
# Purpose: Novelty-based feature transform.
# ######################################################################

import numpy as np

from .base import BaseTransform


class NoveltyTransform(BaseTransform):
    # ##################################################################
    # Author: Cameron M. Gattis
    # Created: 2026-06-14
    # Updated: New.
    # Purpose: Converts novelty scoring methods into
    #          feature-space representations.
    # ##################################################################

    def __init__(self,method):

        self.method = method


    def fit(self,X):

        self.method.fit(X)

        return self


    def forward(self,X):

        transformed = []
        for row in X:
            transformed.append(self.method.explain(row,np.zeros_like(row)))

        return np.asarray(transformed)