from .relu import ReLU
from .sigmoid import Sigmoid
from .tanh import Tanh
from .ensemble import EnsembleActivation
from .grouped import GroupedActivation

# ######################################################################
# Author: Cameron M. Gattis
# Created: 2026-06-13
# Updated: New.
# Purpose: Activation functions and derivatives.
# ######################################################################

import numpy as np
from scipy.special import expit


# ==========================================================
# Sigmoid
# ==========================================================

def sigmoid(x):
    return expit(x)


def sigmoid_derivative(output):
    return output * (1 - output)


# ==========================================================
# ReLU
# ==========================================================

def relu(x):
    return np.maximum(0, x)


def relu_derivative(output):
    return (output > 0).astype(float)


# ==========================================================
# Tanh
# ==========================================================

def tanh(x):
    return np.tanh(x)


def tanh_derivative(output):
    return 1 - (output ** 2)


# ==========================================================
# Registry
# ==========================================================

ACTIVATIONS = {
    "sigmoid": (
        sigmoid,
        sigmoid_derivative
    ),

    "relu": (
        relu,
        relu_derivative
    ),

    "tanh": (
        tanh,
        tanh_derivative
    )
}


def get_activation(name):

    if name not in ACTIVATIONS:
        raise ValueError(
            f"Unknown activation: {name}"
        )

    return ACTIVATIONS[name]