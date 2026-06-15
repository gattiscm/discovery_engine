# ######################################################################
# Author: Cameron M. Gattis
# Created: 2026-06-09
# Updated: New.
# Purpose: Standardized interpretability arrow mappings
#          for statistical and novelty explanations.
# ######################################################################

# ==========================================================
# Relationship Arrows
# ==========================================================

ARROW_SIGNIFICANT_UP = "⇈"
ARROW_UP = "↑"
ARROW_NEUTRAL = "↔"
ARROW_DOWN = "↓"
ARROW_SIGNIFICANT_DOWN = "⇊"
ARROW_UNRELATED = "⇎"
ARROW_MAPS = "↦"

def z_to_arrow(z,minor_cutoff=0.5,major_cutoff=2.0):
    '''
        Converts z-score values into
        interpretability arrows.

        @params z : float

        @return arrow : str
    '''
    # Significant positive deviation
    if z > major_cutoff:
        return (ARROW_SIGNIFICANT_UP)

    # Moderate positive deviation
    elif z > minor_cutoff:
        return (ARROW_UP)

    # Significant negative deviation
    elif z < -major_cutoff:
        return (ARROW_SIGNIFICANT_DOWN)

    # Moderate negative deviation
    elif z < -minor_cutoff:
        return (ARROW_DOWN)

    # Near expected value
    else:
        return (ARROW_NEUTRAL)