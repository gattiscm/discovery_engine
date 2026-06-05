# models/category_memory.py

import numpy as np


class CategoryMemory:

    def __init__(self):
        self.basis = {}

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        classes = np.unique(y)

        for c in classes:
            Xc = X[y == c]

            self.basis[c] = {
                "mean": np.mean(Xc, axis=0),
                "std": np.std(Xc, axis=0)
            }

        return self

    def reconstruct(self, category):
        if category not in self.basis:
            raise ValueError(
                f"Unknown category: {category}"
            )

        return self.basis[category]["mean"]

    def categories(self):
        return list(self.basis.keys())