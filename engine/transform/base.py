# ######################################################################
# Author: Cameron M. Gattis
# Created: 2026-06-13
# Updated: New.
# Purpose: Base interface for feature transforms.
# ######################################################################


class BaseTransform:
    # ##################################################################
    # Author: Cameron M. Gattis
    # Created: 2026-06-13
    # Updated: New.
    # Purpose: Defines the standard interface for
    #          statistical feature transformations.
    # ##################################################################

    def fit(self,X):
        raise NotImplementedError("Subclasses must implement fit().")


    def forward(self,X):
        raise NotImplementedError( "Subclasses must implement forward().")