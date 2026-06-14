# ######################################################################
# Author: Cameron M. Gattis
# Created: 2026-06-13
# Updated: New.
# Purpose: Combines multiple activation functions into a
#          weighted ensemble activation.
# ######################################################################

import numpy as np

from .base import BaseActivation


class EnsembleActivation(BaseActivation):
    # ##################################################################
    # Author: Cameron M. Gattis
    # Created: 2026-06-13
    # Updated: New.
    # Purpose: Aggregates multiple activation functions using
    #          weighted summation.
    # ##################################################################

    def __init__(self,methods):
        '''
            Initializes activation ensemble.

            @params methods : list
                List of tuples:

                    (activation, weight)
        '''
        self.methods = methods
        self.output = None
        self.output = None

    def forward(self,x):
        '''
            Computes weighted activation output.

            @params x : numpy array

            @return output : numpy array
        '''
        output = np.zeros_like(x)
        for method, weight in self.methods:
            output += (weight * method.forward(x))
        self.output = output

        return output

    def backward(self,grad):
        '''
            Computes weighted gradient contribution
            from all activation functions.

            @params grad : numpy array

            @return grad_output : numpy array
        '''
        total = np.zeros_like(x)
        for method, weight in self.methods:
            total += (weight * method.backward(grad))

        return total