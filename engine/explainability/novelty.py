# explainability/novelty.py

import numpy as np


def compute_novelty(
        x,
        reconstruction
):

    residual=(
        x-
        reconstruction
    )

    weighted=(
        np.abs(
            residual
        )
        *
        (
            1+
            np.abs(x)
        )
    )

    return float(
        np.mean(
            weighted
        )
    )


def compute_residual_vector(x, reconstruction):
    x = np.asarray(x, dtype=float)
    reconstruction = np.asarray(reconstruction, dtype=float)

    return np.abs(x - reconstruction)