import numpy as np

from features.base_extractor import (
    BaseExtractor
)


class ShapeExtractor(
        BaseExtractor
):

    def extract(
            self,
            x
    ):

        img=x.reshape(
            16,
            16
        )

        features={}

        features[
            "vertical_center"
        ]=np.mean(
            img[:,8]
        )

        features[
            "horizontal_center"
        ]=np.mean(
            img[8,:]
        )

        features[
            "diag_main"
        ]=np.mean(
            np.diag(
                img
            )
        )

        features[
            "diag_reverse"
        ]=np.mean(
            np.diag(
                np.fliplr(
                    img
                )
            )
        )

        features[
            "top_density"
        ]=np.mean(
            img[:4,:]
        )

        features[
            "bottom_density"
        ]=np.mean(
            img[-4:,:]
        )

        features[
            "left_density"
        ]=np.mean(
            img[:,:4]
        )

        features[
            "right_density"
        ]=np.mean(
            img[:,-4:]
        )

        return features