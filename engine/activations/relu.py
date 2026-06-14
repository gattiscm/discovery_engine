# ######################################################################
# Author: Cameron M. Gattis
# Created: 2026-06-13
# Updated: New.
# Purpose: Rectified Linear Unit (ReLU) activation function.
# ######################################################################

import numpy as np

from .base import BaseActivation


class ReLU( BaseActivation):
    # ##################################################################
    # Author: Cameron M. Gattis
    # Created: 2026-06-13
    # Updated: New.
    # Purpose: Applies ReLU activation and its derivative
    #          during backpropagation.
    # ##################################################################

    def __init__(self):
        self.output = None

    def forward(self,x):
        '''
            Applies ReLU activation.

            @params x : numpy array

            @return output : numpy array
        '''
        return np.maximum(0,x)


    def backward(self,x,grad):
        '''
            Applies ReLU derivative.

            @params grad : numpy array

            @return grad_output : numpy array
        '''
        return (grad * (x > 0))