# ######################################################################
# Author: Cameron M. Gattis
# Created: 2026-06-13
# Updated: New.
# Purpose: Base interface for neural network layers.
# ######################################################################


class Layer:
    # ##################################################################
    # Author: Cameron M. Gattis
    # Created: 2026-06-13
    # Updated: New.
    # Purpose: Defines the standard forward and backward
    #          propagation interface for all layers.
    # ##################################################################

    def forward(self, X):
        '''
            Performs forward propagation.

            @params X : numpy array

            @return output : numpy array

            @exception NotImplementedError :
                Raised if subclass does not implement.
        '''
        raise NotImplementedError


    def backward(self, grad):
        '''
            Performs backward propagation.

            @params grad : numpy array
                Gradient from next layer.

            @return gradient : numpy array

            @exception NotImplementedError :
                Raised if subclass does not implement.
        '''
        raise NotImplementedError