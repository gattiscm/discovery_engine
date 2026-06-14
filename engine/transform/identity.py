# ######################################################################
# Author: Cameron M. Gattis
# Created: 2026-06-13
# Updated: New.
# Purpose: Identity transform.
# ######################################################################

from .base import BaseTransform


class IdentityTransform(BaseTransform):

    def fit(self,X):
        pass


    def forward(self,X):
        return X