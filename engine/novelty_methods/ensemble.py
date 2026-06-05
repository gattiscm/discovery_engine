import numpy as np

from .base import BaseNovelty


class EnsembleNovelty(
        BaseNovelty
):

    def __init__(
            self,
            methods
    ):

        self.methods = methods


    def fit(
            self,
            X
    ):

        for method, weight in self.methods:

            method.fit(
                X
            )


    def compute(
            self,
            x,
            reconstruction
    ):

        total = 0.0

        for method, weight in self.methods:

            score = method.compute(
                x,
                reconstruction
            )

            total += (
                weight *
                score
            )

        return total


    def explain(
            self,
            x,
            reconstruction
    ):

        explanation = {}

        for method, weight in self.methods:

            name = (
                method.__class__.__name__
            )

            explanation[name] = {

                "weight":
                    weight,

                "score":
                    method.compute(
                        x,
                        reconstruction
                    )
            }

        return explanation