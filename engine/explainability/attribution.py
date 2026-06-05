# explainability/attribution.py

import numpy as np


def compute_feature_contributions(
        x,
        reconstruction
):

    x = np.asarray(
        x,
        dtype=float
    )

    reconstruction = np.asarray(
        reconstruction,
        dtype=float
    )

    # =============================
    # preserve sign
    # =============================

    signed_residual = (
        x -
        reconstruction
    )

    magnitude = np.abs(
        signed_residual
    )

    total = np.sum(
        magnitude
    )

    if total == 0:
        total = 1

    contribution = (
        magnitude
        /
        total
    )

    direction = np.sign(
        signed_residual
    )

    return {

        "magnitude":
            contribution,

        "direction":
            direction,

        "signed":
            signed_residual
    }


def compute_group_contributions(
        feature_contributions,
        groups
):

    contributions = {}

    max_idx = (
        len(
            feature_contributions
        ) - 1
    )

    for group_name, indices in (
            groups.items()
    ):

        valid = [

            i

            for i in indices

            if i <= max_idx

        ]

        if len(valid) == 0:

            contributions[
                group_name
            ] = 0.0

            continue

        value = np.sum(

            feature_contributions[
                valid
            ]

        )

        contributions[
            group_name
        ] = float(
            value
        )

    total = sum(
        contributions.values()
    )

    if total > 0:

        contributions = {

            k: (
                v / total
            )

            for k, v in
            contributions.items()
        }

    return contributions