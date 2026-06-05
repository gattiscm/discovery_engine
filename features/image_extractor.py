# features/image_extractor.py

import numpy as np

from features.base_extractor import (
    BaseExtractor
)


class ImageExtractor(
        BaseExtractor
):

    def __init__(
            self,
            bands=4,
            eps=1e-8
    ):

        self.bands = bands
        self.eps = eps

        self.output_feature_names = []

    def fit(
            self,
            X,
            feature_names=None
    ):

        self.output_feature_names = [

            "global_mean",
            "global_std",
            "global_energy",

            "vertical_symmetry",
            "horizontal_symmetry",

            "center_density"
        ]

        for i in range(
                self.bands
        ):

            self.output_feature_names.append(
                f"row_band_{i}"
            )

        for i in range(
                self.bands
        ):

            self.output_feature_names.append(
                f"column_band_{i}"
            )

        self.output_feature_names.extend([

            "quadrant_1",
            "quadrant_2",
            "quadrant_3",
            "quadrant_4"

        ])

        return self

    def get_feature_names(
            self
    ):

        return self.output_feature_names

    def extract(
            self,
            x
    ):

        img = np.asarray(
            x,
            dtype=float
        )

        # ----------------------------------
        # Accept:
        #
        # (64,)      digits
        # (256,)     shapes
        # (784,)     fashion
        # (h,w)
        #
        # ----------------------------------

        if img.ndim == 1:

            side = int(
                np.sqrt(
                    len(img)
                )
            )

            img = img.reshape(
                side,
                side
            )

        h, w = img.shape

        features = {}

        # ==================================
        # Global
        # ==================================

        features[
            "global_mean"
        ] = np.mean(
            img
        )

        features[
            "global_std"
        ] = np.std(
            img
        )

        features[
            "global_energy"
        ] = np.mean(
            img ** 2
        )

        # ==================================
        # Symmetry
        # ==================================

        flipped_lr = np.fliplr(
            img
        )

        flipped_ud = np.flipud(
            img
        )

        features[
            "vertical_symmetry"
        ] = 1.0 / (
                np.mean(
                    np.abs(
                        img -
                        flipped_lr
                    )
                )
                +
                self.eps
        )

        features[
            "horizontal_symmetry"
        ] = 1.0 / (
                np.mean(
                    np.abs(
                        img -
                        flipped_ud
                    )
                )
                +
                self.eps
        )

        # ==================================
        # Center density
        # ==================================

        center = img[
                 h // 4: 3 * h // 4,
                 w // 4: 3 * w // 4
                 ]

        features[
            "center_density"
        ] = np.mean(
            center
        )

        # ==================================
        # Row bands
        # ==================================

        row_edges = np.linspace(
            0,
            h,
            self.bands + 1,
            dtype=int
        )

        for i in range(
                self.bands
        ):

            band = img[
                   row_edges[i]:
                   row_edges[i + 1],
                   :
                   ]

            features[
                f"row_band_{i}"
            ] = np.mean(
                band
            )

        # ==================================
        # Column bands
        # ==================================

        col_edges = np.linspace(
            0,
            w,
            self.bands + 1,
            dtype=int
        )

        for i in range(
                self.bands
        ):

            band = img[
                   :,
                   col_edges[i]:
                   col_edges[i + 1]
                   ]

            features[
                f"column_band_{i}"
            ] = np.mean(
                band
            )

        # ==================================
        # Quadrants
        # ==================================

        features[
            "quadrant_1"
        ] = np.mean(
            img[
            :h // 2,
            :w // 2
            ]
        )

        features[
            "quadrant_2"
        ] = np.mean(
            img[
            :h // 2,
            w // 2:
            ]
        )

        features[
            "quadrant_3"
        ] = np.mean(
            img[
            h // 2:,
            :w // 2
            ]
        )

        features[
            "quadrant_4"
        ] = np.mean(
            img[
            h // 2:,
            w // 2:
            ]
        )

        return features