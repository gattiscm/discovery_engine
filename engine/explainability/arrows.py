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

def z_to_arrow(z):
    '''
        Converts z-score values into
        interpretability arrows.

        @params z : float

        @return arrow : str
    '''
    # Significant positive deviation
    if z > 2:
        return (ARROW_SIGNIFICANT_UP)

    # Moderate positive deviation
    elif z > 0.5:
        return (ARROW_UP)

    # Significant negative deviation
    elif z < -2:
        return (ARROW_SIGNIFICANT_DOWN)

    # Moderate negative deviation
    elif z < -0.5:
        return (ARROW_DOWN)

    # Near expected value
    else:
        return (ARROW_NEUTRAL)