import math


def next_power_two_latent(
        input_dims,
        multiplier=2
):

    target = (
        input_dims *
        multiplier
    )

    latent = 1

    while latent <= target:
        latent *= 2

    return latent

def auto_latent_units(
        input_dim,
        multiplier=2,
        max_units=1024
):

    target = input_dim * multiplier

    units = 1

    while units < target:
        units *= 2

    return min(
        units,
        max_units
    )