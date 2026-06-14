# ######################################################################
# Author: Cameron M. Gattis
# Created: 2026-06-13
# Updated: New.
# Purpose: Hyperbolic tangent activation function.
# ######################################################################

import numpy as np

from .base import BaseActivation


class Tanh(BaseActivation):
    # ##################################################################
    # Author: Cameron M. Gattis
    # Created: 2026-06-13
    # Updated: New.
    # Purpose: Applies hyperbolic tangent activation and its
    #          derivative during backpropagation.
    # ##################################################################

    def __init__(self):
        '''
            Initializes tanh activation.
        '''

        self.output = None


    def forward(self,x):
        '''
            Applies tanh activation.

            @params x : numpy array

            @return output : numpy array
        '''

        self.output = np.tanh(x)

        return self.output


    def backward(self,grad):
        '''
            Applies tanh derivative.

            @params grad : numpy array

            @return grad_output : numpy array
        '''

        return (grad * (1 - self.output ** 2))