# ######################################################################
# Author: Cameron M. Gattis
# Created: 2026-06-05
# Updated: New.
# Purpose: Default use class for initialization.
# ######################################################################

class BaseExtractor:
    # ######################################################################
    # Author: Cameron M. Gattis
    # Created: 2026-06-05
    # Updated: New.
    # Purpose: Placeholder class that is over written by individual
    #          extractors as needed.
    # ######################################################################
    def extract(self,x):
        '''
            Base extraction feature, not used for production.

            @exception NotImplementedError : none
        '''

        raise NotImplementedError