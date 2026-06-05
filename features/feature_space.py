# engine/feature_space.py

import numpy as np


class FeatureSpace:

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)

        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)

        self.std[self.std == 0] = 1.0

        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)

        if self.mean is None or self.std is None:
            raise ValueError("FeatureSpace must be fit before transform().")

        return (X - self.mean) / self.std

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)