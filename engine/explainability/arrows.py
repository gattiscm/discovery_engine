ARROW_SIGNIFICANT_UP = "⇈"
ARROW_UP = "↑"
ARROW_NEUTRAL = "↔"
ARROW_DOWN = "↓"
ARROW_SIGNIFICANT_DOWN = "⇊"
ARROW_UNRELATED = "⇎"
ARROW_MAPS = "↦"

def z_to_arrow(z):
    if z > 2:
        return ARROW_SIGNIFICANT_UP
    elif z > 0.5:
        return ARROW_UP
    elif z < -2:
        return ARROW_SIGNIFICANT_DOWN
    elif z < -0.5:
        return ARROW_DOWN
    else:
        return ARROW_NEUTRAL