# ######################################################################
# Author: Cameron M. Gattis
# Created: 2026-06-13
# Updated: New.
# Purpose: Applies activation functions independently to
#          predefined feature groups.
# ######################################################################

import numpy as np

from .base import BaseActivation
from copy import deepcopy


class GroupedActivation(BaseActivation):
    # ##################################################################
    # Author: Cameron M. Gattis
    # Created: 2026-06-13
    # Updated: New.
    # Purpose: Applies a shared activation function across
    #          multiple feature groups.
    # ##################################################################

    def __init__(self,groups,activation):
        '''
            Initializes grouped activation.

            @params groups : list
                Collection of feature index groups.

                    activation : BaseActivation
                Activation applied to each group.
        '''
        self.groups = groups
        self.activations = [
            deepcopy(activation)
            for _ in groups
        ]
        self.output = None

    def forward(self,x):
        '''
            Applies activation independently to
            each feature group.

            @params x : numpy array

            @return output : numpy array
        '''
        output = np.zeros_like(x)
        for group, activation in zip(self.groups,self.activations):
            output[:, group] = (activation.forward(x[:, group] ))

        return output

    def backward(self,grad):
        '''
            Applies activation gradients independently
            to each feature group.

            @params grad : numpy array

            @return grad_output : numpy array
        '''
        output = np.zeros_like(grad)

        for group, activation in zip(self.groups,self.activations):
            output[:, group] = (activation.backward(grad[:, group]))

        return output