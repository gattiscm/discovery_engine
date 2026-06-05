

class BaseExtractor:

    def extract(
            self,
            x
    ):
        raise NotImplementedError

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

        return {

            "vertical_center":
                np.mean(
                    img[:,8]
                ),

            "horizontal_center":
                np.mean(
                    img[8,:]
                ),

            "main_diagonal":
                np.mean(
                    np.diag(img)
                ),

            "reverse_diagonal":
                np.mean(
                    np.diag(
                        np.fliplr(img)
                    )
                ),

            "top_edge":
                np.mean(
                    img[0,:]
                ),

            "bottom_edge":
                np.mean(
                    img[-1,:]
                )
        }