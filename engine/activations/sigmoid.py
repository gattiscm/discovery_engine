# ######################################################################
# Author: Cameron M. Gattis
# Created: 2026-06-13
# Updated: New.
# Purpose: Sigmoid activation function.
# ######################################################################

import numpy as np

from .base import BaseActivation


class Sigmoid(BaseActivation):
    # ##################################################################
    # Author: Cameron M. Gattis
    # Created: 2026-06-13
    # Updated: New.
    # Purpose: Applies sigmoid activation and its derivative
    #          during backpropagation.
    # ##################################################################

    def __init__(self):

        self.output = None


    def forward(self,x):
        '''
            Applies sigmoid activation.

            @params x : numpy array

            @return output : numpy array
        '''

        self.output = (1.0 / (1.0 + np.exp(-x)))

        return self.output


    def backward(self,grad):
        '''
            Applies sigmoid derivative.

            @params grad : numpy array

            @return grad_output : numpy array
        '''

        return (grad * self.output * (1 - self.output))