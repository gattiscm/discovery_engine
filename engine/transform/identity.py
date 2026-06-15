# engine/transforms/identity.py

from .base import BaseTransform


class IdentityTransform(BaseTransform):
    # ##################################################################
    # Author: Cameron M. Gattis
    # Created: 2026-06-14
    # Updated: New.
    # Purpose: Pass-through transform that returns input data
    #          unchanged.
    # ##################################################################

    def fit(self, X):
        '''
            No fitting required.

            @params X : numpy array

            @return self : IdentityTransform
        '''
        return self


    def forward(self, X):
        '''
            Returns input data unchanged.

            @params X : numpy array

            @return X : numpy array
        '''
        return X