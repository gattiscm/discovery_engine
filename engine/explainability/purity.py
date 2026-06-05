# explainability/purity.py

import numpy as np


def compute_purity(novelty_score):
    novelty_score = float(novelty_score)

    purity = 1.0 / (1.0 + novelty_score)

    return float(purity)