# ######################################################################
# Author: Cameron M. Gattis
# Created: 2026-06-13
# Updated: New.
# Purpose: Utility functions for combining outputs from
#          multiple layer branches.
# ######################################################################

import numpy as np


def merge_outputs(outputs, merge="concat") -> np.ndarray:
    '''
        Merges outputs from multiple branches.

        Supported methods:

            concat
                Concatenates outputs along the
                feature dimension.

            add
                Computes element-wise sum.

            multiply
                Computes element-wise product.

        @params outputs : list
            List of numpy arrays.

                merge : str
            Merge strategy.

        @return output : numpy array

        @exception ValueError :
            Raised if no outputs are provided or
            merge method is unsupported.
    '''

    if len(outputs) == 0:
        raise ValueError("No outputs provided to merge.")

    if merge == "concat":
        # Stack features side-by-side
        return np.concatenate(outputs, axis=1)

    if merge == "add":
        # Element-wise summation
        return np.sum(outputs, axis=0)

    if merge == "multiply":
        # Element-wise summation
        result = outputs[0]
        for out in outputs[1:]:
            result = result * out

        return result

    raise ValueError(
        f"Unknown merge method: {merge}"
    )