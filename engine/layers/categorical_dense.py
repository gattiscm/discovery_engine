# layers/categorical_dense.py

import numpy as np

from engine.layers.base_layer import Layer
from engine.layers.dense import Dense
from engine.layers.merge import merge_outputs


class CategoricalDense(Layer):

    def __init__(
            self,
            units,
            groups=None,
            mode="parallel",
            merge="concat",
            seed=42
    ):
        self.units = units
        self.groups = groups
        self.mode = mode
        self.merge = merge
        self.seed = seed

        self.branches = {}
        self.built = False

    def build(self, input_dim):

        if self.groups is None:
            self.groups = {
                "default": list(
                    range(input_dim)
                )
            }

        carry_size = 0

        for i, (name, indices) in enumerate(
                self.groups.items()
        ):

            if self.mode == "sequential":

                input_size = (
                        len(indices)
                        +
                        carry_size
                )

                carry_size = self.units

            else:

                input_size = len(indices)

            self.branches[name] = Dense(

                inputs=input_size,

                outputs=self.units,

                seed=self.seed + i
            )

        self.built = True

    def forward(self, X):
        X = np.asarray(X, dtype=float)

        if not self.built:
            self.build(X.shape[1])

        if self.mode == "parallel":
            return self._forward_parallel(X)

        if self.mode == "sequential":
            return self._forward_sequential(X)

        raise ValueError(
            f"Unknown CategoricalDense mode: {self.mode}"
        )

    def _forward_parallel(self, X):
        outputs = []

        for name, indices in self.groups.items():
            X_group = X[:, indices]

            out = self.branches[name].forward(
                X_group
            )

            outputs.append(out)

        return merge_outputs(
            outputs,
            merge=self.merge
        )

    def _forward_sequential(self, X):

        carry = None

        outputs = []

        for name, indices in self.groups.items():

            X_group = X[:, indices]

            if carry is not None:
                X_group = np.concatenate(
                    [
                        X_group,
                        carry
                    ],
                    axis=1
                )

            out = self.branches[name].forward(
                X_group
            )

            carry = out

            outputs.append(out)

        return merge_outputs(
            outputs,
            merge=self.merge
        )

    def backward(
            self,
            grad
    ):

        branch_names = list(
            self.groups.keys()
        )

        # split gradient for concat mode
        if self.merge == "concat":

            grad_splits = np.split(
                grad,
                len(branch_names),
                axis=1
            )

        else:

            grad_splits = [

                grad for _ in branch_names

            ]

        grad_inputs = []

        for name, branch_grad in zip(

                reversed(branch_names),

                reversed(
                    grad_splits
                )
        ):
            branch = (
                self.branches[name]
            )

            grad_input = (

                branch.backward(
                    branch_grad
                )
            )

            grad_inputs.append(
                grad_input
            )

        return np.concatenate(

            grad_inputs,

            axis=1
        )