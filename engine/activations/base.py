# ######################################################################
# Author: Cameron M. Gattis
# Created: 2026-06-13
# Updated: New.
# Purpose: Base interface for activation functions.
# ######################################################################


class BaseActivation:
    # ##################################################################
    # Author: Cameron M. Gattis
    # Created: 2026-06-13
    # Updated: New.
    # Purpose: Defines the standard activation interface used
    #          throughout the engine.
    # ##################################################################

    def forward(self,x):
        '''
            Applies activation function.

            @params x : numpy array

            @return output : numpy array

            @exception NotImplementedError :
                Raised if subclass does not implement.
        '''
        raise NotImplementedError("Subclasses must implement forward().")


    def backward(self,x,grad):
        '''
            Applies activation derivative during
            backpropagation.

            @params x : numpy array
                Forward activation output.

                    grad : numpy array
                Incoming gradient.

            @return grad_output : numpy array

            @exception NotImplementedError :
                Raised if subclass does not implement.
        '''
        raise NotImplementedError("Subclasses must implement backward().")