# ######################################################################
# Author: Cameron M. Gattis
# Created: 2026-06-06
# Updated: New.
# Purpose: Converts mixed-type tabular data into a numeric feature
#          matrix for DiscoveryModel, including numeric values,
#          z-score features, and one-hot categorical features.
# ######################################################################

import numpy as np


class FeatureSpace:
    # ######################################################################
    # Author: Cameron M. Gattis
    # Created: 2026-06-06
    # Updated: New.
    # Purpose: Converts mixed-type tabular data into a numeric feature
    #          matrix for DiscoveryModel, including numeric values,
    #          z-score features, and one-hot categorical features.
    # ######################################################################

    def __init__(self):
        '''
            Object initializer.
        '''

        self.mean = None
        self.std = None

    def fit(self, X):
        '''
            Method to fit and learn numerical and categorical vocabularies.

            @params X : dataframe

            @return self : object
        '''

        X = np.asarray(X, dtype=float)

        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)

        self.std[self.std == 0] = 1.0

        return self

    def transform(self, X):
        '''
            Converter for pandas dataframe by row into a numeric matrix.

            @params X : dataframe

            @return array : numpy array

            @exception ValueError : none
        '''

        X = np.asarray(X, dtype=float)

        if self.mean is None or self.std is None:
            raise ValueError("FeatureSpace must be fit before transform().")

        return (X - self.mean) / self.std

    def fit_transform(self, X):
        '''
            Fits the extractor and transforms the data.

            @params X : pandas dataframe

            @return self : pandas dataframe
        '''

        self.fit(X)
        return self.transform(X)