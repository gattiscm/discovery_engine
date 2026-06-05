import numpy as np
import pandas as pd


class TabularExtractor:

    def __init__(
            self,
            eps=1e-8
    ):

        self.eps = eps

        self.numeric_columns = []
        self.categorical_columns = []

        self.mean_ = {}
        self.std_ = {}

        self.categories_ = {}

        self.feature_names = []
        self.output_feature_names = []

    def fit(
            self,
            X
    ):

        if not isinstance(
                X,
                pd.DataFrame
        ):

            X = pd.DataFrame(
                X
            )

        self.numeric_columns = list(

            X.select_dtypes(
                include=[
                    np.number
                ]
            ).columns

        )

        self.categorical_columns = list(

            X.select_dtypes(
                exclude=[
                    np.number
                ]
            ).columns

        )

        for col in self.numeric_columns:

            values = X[col].astype(
                float
            )

            self.mean_[col] = float(
                values.mean()
            )

            std = float(
                values.std()
            )

            if std < self.eps:
                std = 1.0

            self.std_[col] = std

        for col in self.categorical_columns:

            categories = sorted(

                X[col]
                .astype(str)
                .unique()
                .tolist()

            )

            self.categories_[col] = (
                categories
            )

        self.output_feature_names = []

        #
        # numeric
        #

        for col in self.numeric_columns:

            self.output_feature_names.append(
                col
            )

            self.output_feature_names.append(
                f"{col}_z"
            )

        #
        # one-hot
        #

        for col in self.categorical_columns:

            for category in self.categories_[
                col
            ]:

                self.output_feature_names.append(

                    f"{col}_{category}"

                )

        return self

    def get_feature_names(
            self
    ):

        return list(
            self.output_feature_names
        )

    def transform(
            self,
            X
    ):

        if not isinstance(
                X,
                pd.DataFrame
        ):

            X = pd.DataFrame(
                X
            )

        rows = []

        for _, row in X.iterrows():

            rows.append(

                list(

                    self.extract(
                        row
                    ).values()

                )

            )

        return np.asarray(
            rows,
            dtype=float
        )

    def fit_transform(
            self,
            X
    ):

        self.fit(
            X
        )

        return self.transform(
            X
        )

    def extract(
            self,
            row
    ):

        if isinstance(
                row,
                pd.Series
        ):

            row = row.to_dict()

        features = {}

        #
        # numeric
        #

        for col in self.numeric_columns:

            value = float(
                row[col]
            )

            z = (

                value

                -

                self.mean_[col]

            ) / (

                self.std_[col]

                +

                self.eps

            )

            features[col] = value

            features[
                f"{col}_z"
            ] = z

        #
        # one-hot
        #

        for col in self.categorical_columns:

            current = str(
                row[col]
            )

            for category in self.categories_[
                col
            ]:

                features[
                    f"{col}_{category}"
                ] = (

                    1.0

                    if current == category

                    else 0.0

                )

        return features