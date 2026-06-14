# ######################################################################
# Author: Cameron M. Gattis
# Created: 2026-06-13
# Updated: New.
# Purpose: Dense layer that processes user-defined feature groups
#          independently or sequentially before merging outputs.
# Parallel Mode:
#   Group outputs are computed independently.
#
# Sequential Mode:
#   Each group receives its own features plus the
#   previous group's output.
# ######################################################################


import numpy as np

from engine.layers.base_layer import Layer
from engine.layers.dense import Dense
from engine.layers.merge import merge_outputs


class CategoricalDense(Layer):
    # ##################################################################
    # Author: Cameron M. Gattis
    # Created: 2026-06-13
    # Updated: New.
    # Purpose: Applies separate Dense branches to feature groups
    #          and merges branch outputs into a single tensor.
    # ##################################################################

    def __init__(
            self,
            units,
            groups=None,
            mode="parallel",
            merge="concat",
            seed=42
    ):
        '''
            Initializes grouped dense layer.

            @params units : int
                    Output units per branch.

                groups : dict
                    Mapping of group names to feature indices.

                mode : str
                    Branch execution mode:
                        "parallel"
                        "sequential"

                merge : str
                    Output merge strategy.

                seed : int
                    Random seed.
        '''
        self.units = units
        self.groups = groups
        self.mode = mode
        self.merge = merge
        self.seed = seed

        self.branches = {}
        self.built = False

    def build(self, input_dim) -> int:
        '''
            Creates Dense branches for each feature group.

            @params input_dim : int
                Total number of input features.
        '''
        if self.groups is None:
            self.groups = {"default": list(range(input_dim))}

        carry_size = 0

        for i, (name, indices) in enumerate(self.groups.items()):
            if self.mode == "sequential":
                input_size = (len(indices) + carry_size)
                carry_size = self.units

            else:
                input_size = len(indices)

            self.branches[name] = Dense(
                inputs=input_size,
                outputs=self.units,
                seed=self.seed + i
            )

        self.built = True

    def forward(self, X) -> np.ndarray:
        '''
            Executes grouped forward propagation.

            @params X : numpy array

            @return output : numpy array

            @exception ValueError :
                Raised on unsupported mode.
        '''
        X = np.asarray(X, dtype=float)
        if not self.built:
            self.build(X.shape[1])
        if self.mode == "parallel":
            return self._forward_parallel(X)
        if self.mode == "sequential":
            return self._forward_sequential(X)
        raise ValueError(f"Unknown CategoricalDense mode: {self.mode}")

    def _forward_parallel(self, X) -> np.ndarray :
        '''
                Processes all groups independently.

                @params X : numpy array

                @return output : numpy array
            '''
        outputs = []

        for name, indices in self.groups.items():
            X_group = X[:, indices]
            out = self.branches[name].forward(X_group)
            outputs.append(out)

        return merge_outputs(outputs,merge=self.merge)

    def _forward_sequential(self, X) -> np.ndarray:
        '''
            Processes groups sequentially, passing each
            branch output to the next branch as carry data.

            @params X : numpy array

            @return output : numpy array
        '''

        carry = None
        outputs = []

        for name, indices in self.groups.items():
            X_group = X[:, indices]
            if carry is not None:
                X_group = np.concatenate([X_group,carry],axis=1)

            out = self.branches[name].forward(X_group)
            carry = out
            outputs.append(out)

        return merge_outputs(outputs,merge=self.merge)

    def backward(self,grad) -> np.ndarray:
        '''
            Performs backpropagation through all branches.

            @params grad : numpy array

            @return grad_input : numpy array
        '''

        branch_names = list( self.groups.keys())
        # split gradient for concat mode
        if self.merge == "concat":
            grad_splits = np.split(grad,len(branch_names),axis=1)

        else:
            grad_splits = [grad for _ in branch_names]

        grad_inputs = []

        for name, branch_grad in zip(reversed(branch_names),reversed(grad_splits)):
            branch = (self.branches[name])
            grad_input = (branch.backward(branch_grad))
            grad_inputs.append(grad_input)

        return np.concatenate(grad_inputs,axis=1)