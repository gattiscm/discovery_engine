# ######################################################################
# Author: Cameron M. Gattis
# Created: 2026-06-05
# Updated: New.
# Purpose: Converts mixed-type tabular data into a numeric feature
#          matrix for DiscoveryModel, including numeric values,
#          z-score features, and one-hot categorical features.
# ######################################################################

import numpy as np
import pandas as pd

class TabularExtractor:
    # ######################################################################
    # Author: Cameron M. Gattis
    # Created: 2026-06-05
    # Updated: New.
    # Purpose: Collection of methods to process tabular data.
    # ######################################################################

    def __init__(self,eps=1e-8):
        '''
            Function initializer.
        '''

        self.eps = eps
        self.numeric_columns = []
        self.categorical_columns = []
        self.mean_ = {}
        self.std_ = {}
        self.categories_ = {}
        self.feature_names = []
        self.output_feature_names = []

    def fit(self,X):
        '''
            Method to fit and learn numerical and categorical vocabularies.

            @params X : dataframe

            @return self : object
        '''

        if not isinstance(X,pd.DataFrame):
            X = pd.DataFrame(X)

        self.numeric_columns = list(X.select_dtypes(include=[np.number]).columns)

        self.categorical_columns = list(X.select_dtypes(exclude=[np.number]).columns)

        for col in self.numeric_columns:
            values = X[col].astype(float)
            self.mean_[col] = float(values.mean())
            std = float(values.std(ddof=0))
            if std < self.eps:
                std = 1.0

            self.std_[col] = std

        for col in self.categorical_columns:
            categories = sorted(X[col].astype(str).unique().tolist())
            self.categories_[col] = (categories)

        self.output_feature_names = []
        #
        # numeric
        #
        for col in self.numeric_columns:
            self.output_feature_names.append(col)
            self.output_feature_names.append(f"{col}_z")
        #
        # one-hot
        #
        for col in self.categorical_columns:
            for category in self.categories_[col]:
                self.output_feature_names.append(f"{col}_{category}")

        return self

    def get_feature_names(self):
        '''
            Generates feature names after expansions.

            @params na : na

            @return list : list
        '''

        return list(self.output_feature_names)

    def transform(self,X):
        '''
            Converter for pandas dataframe by row into a numeric matrix.

            @params X : dataframe

            @return array : numpy array
        '''

        if not isinstance(X,pd.DataFrame):
            X = pd.DataFrame(X)

        rows = []

        for _, row in X.iterrows():
            rows.append(list(self.extract(row).values()) )

        return np.asarray(rows,dtype=float)

    def fit_transform(self,X):
        '''
            Fits the extractor and transforms the data.

            @params X : pandas dataframe

            @return self : pandas dataframe
        '''

        self.fit(X)

        return self.transform(X)

    def extract(self,row):
        '''
            Method to convert a dataframe row into a numeric feature dictonary.

            @params row : pandas dataframe row

            @return features : dictionary
        '''

        if isinstance(row,pd.Series):
            row = row.to_dict()
        features = {}
        #
        # numeric
        #
        for col in self.numeric_columns:
            value = float(row[col])
            z = (value - self.mean_[col]) / (self.std_[col] +  self.eps)
            features[col] = value
            features[f"{col}_z"] = z
        #
        # one-hot
        #
        for col in self.categorical_columns:
            current = str(row[col])
            for category in self.categories_[col]:
                features[f"{col}_{category}"] = (1.0
                    if current == category
                    else 0.0)
        return features