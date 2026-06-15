# ######################################################################
# Author: Cameron M. Gattis
# Created: 2026-06-09
# Updated: New.
# Purpose: Computes feature-level and group-level
#          attribution scores from reconstruction
#          residuals.
# ######################################################################

import numpy as np


def compute_feature_contributions(x,reconstruction):
    '''
        Computes feature attribution information
        from reconstruction residuals.

        @params x : numpy array
            Original sample.

                reconstruction : numpy array
            Model reconstruction.

        @return contributions : dict
            Contains normalized contribution
            magnitude, direction, and signed
            residual values.
    '''

    x = np.asarray(x,dtype=float)
    reconstruction = np.asarray(reconstruction,dtype=float)

    # Compute signed reconstruction error
    signed_residual = (x - reconstruction)

    # Magnitude of reconstruction error
    magnitude = np.abs(signed_residual)
    total = np.sum(magnitude)

    # Prevent division by zero
    if total == 0:
        total = 1

    # Normalize feature contributions
    contribution = (magnitude / total)

    # Track residual direction
    direction = np.sign(signed_residual)

    return {
        "magnitude": contribution,
        "direction": direction,
        "signed": signed_residual
    }


def compute_group_contributions(feature_contributions,groups):
    '''
        Aggregates feature contributions into
        predefined feature groups.

        @params feature_contributions : numpy array
            Feature attribution magnitudes.

                groups : dict
            Mapping of group names to feature
            indices.

        @return contributions : dict
            Normalized contribution per group.
    '''

    contributions = {}
    max_idx = (len(feature_contributions) - 1)

    for group_name, indices in (groups.items()):
        # Ignore invalid indices
        valid = [i for i in indices if i <= max_idx]

        if len(valid) == 0:
            contributions[group_name] = 0.0
            continue

        value = np.sum(feature_contributions[valid])
        contributions[group_name] = float(value)
    total = sum(contributions.values())

    # Normalize group contributions
    if total > 0:
        contributions = {k: (v / total) for k, v in (contributions.items())}

    return contributions