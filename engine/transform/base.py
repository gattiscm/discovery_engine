# engine/transforms/base.py

class BaseTransform:
    # ##################################################################
    # Author: Cameron M. Gattis
    # Created: 2026-06-14
    # Updated: New.
    # Purpose: Defines the standard interface for feature
    #          transformation methods.
    # ##################################################################

    def fit(self, X):
        '''
            Learns required statistics from training data.

            @params X : numpy array

            @return self : BaseTransform
        '''
        raise NotImplementedError("Subclasses must implement fit().")


    def forward(self, X):
        '''
            Applies feature transformation.

            @params X : numpy array

            @return transformed : numpy array
        '''
        raise NotImplementedError("Subclasses must implement forward().")