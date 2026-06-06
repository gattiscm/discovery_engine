# ######################################################################
# Author: Cameron M. Gattis
# Created: 2026-06-06
# Updated: New.
# Purpose: Generator to calculate the number of latent units to build
#          for an automatic model.
# ######################################################################

def auto_latent_units(
        input_dim,
        multiplier=2,
        max_units=1024
):
    '''
        Utilizes dimensional number to calculate an automated number
        of latent units for latent mapping.

        @params input_dim : int
                multiplier : KWARG, int
                max_units : KWaRG, int

        @return calculated : int
    '''

    target = input_dim * multiplier

    units = 1

    while units < target:
        units *= 2

    return min(units,max_units)