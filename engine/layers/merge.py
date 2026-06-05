# layers/merge.py

import numpy as np


def merge_outputs(outputs, merge="concat"):

    if len(outputs) == 0:
        raise ValueError("No outputs provided to merge.")

    if merge == "concat":
        return np.concatenate(outputs, axis=1)

    if merge == "add":
        return np.sum(outputs, axis=0)

    if merge == "multiply":
        result = outputs[0]

        for out in outputs[1:]:
            result = result * out

        return result

    raise ValueError(
        f"Unknown merge method: {merge}"
    )