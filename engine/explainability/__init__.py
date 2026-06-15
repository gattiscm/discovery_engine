# ######################################################################
# Author: Cameron M. Gattis
# Created: 2026-06-14
# Updated: New.
# Purpose: Explainability utilities for latent analysis,
#          attribution, novelty interpretation, and
#          statistical visualization.
# ######################################################################

from .arrows import (
    ARROW_SIGNIFICANT_UP,
    ARROW_UP,
    ARROW_NEUTRAL,
    ARROW_DOWN,
    ARROW_SIGNIFICANT_DOWN,
    ARROW_UNRELATED,
    ARROW_MAPS,
    z_to_arrow
)

from .attribution import (
    compute_feature_contributions,
    compute_group_contributions
)

from .latent_map import (
    build_latent_map
)

from .novelty import (
    compute_novelty,
    compute_residual_vector
)