# ######################################################################
# Author: Cameron M. Gattis
# Created: 2026-06-13
# Updated: New.
# Purpose: Base interface for novelty scoring methods.
# ######################################################################


class BaseNovelty:
    # ##################################################################
    # Author: Cameron M. Gattis
    # Created: 2026-06-13
    # Updated: New.
    # Purpose: Defines the standard interface used by all
    #          novelty scoring implementations.
    # ##################################################################

    def fit(self, X):
        '''
            Learns any required statistics from
            training data.

            @params X : numpy array
        '''
        pass


    def compute(self, x, reconstruction):
        '''
            Computes novelty score.

            @params x : numpy array
                    reconstruction : numpy array

            @return novelty : float
        '''
        pass


    def explain(self, x, reconstruction):
        '''
            Returns feature-level explanation of
            novelty score.

            @params x : numpy array
                    reconstruction : numpy array

            @return explanation : object
        '''
        pass