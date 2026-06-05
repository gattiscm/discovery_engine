import os
import pandas as pd
import numpy as np


import os
import pandas as pd
import numpy as np


class NovelMemory:

    def __init__(
            self,
            path="results/novel_memory.csv"
    ):

        self.path=path

        if not os.path.exists(
                path
        ):

            starter=pd.DataFrame(

                columns=[

                    "novelty",
                    "closest"

                ]

            )

            starter.to_csv(

                path,
                index=False

            )

    def store(

            self,
            latent,
            novelty,
            closest,
            feature_contrib=None

    ):

        latent = np.asarray(
            latent
        ).flatten()

        row = {

            "novelty":
                float(novelty),

            "closest": str(closest)

        }

        for i, v in enumerate(
                latent
        ):
            row[
                f"latent_{i}"
            ] = float(v)


    def load(self):

        try:

            return pd.read_csv(
                self.path
            )

        except:

            return pd.DataFrame()