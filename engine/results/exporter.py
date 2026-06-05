import json
from pathlib import Path
import numpy as np
import pandas as pd


RESULTS_DIR = Path(
    "results"
)

RESULTS_DIR.mkdir(
    exist_ok=True
)

def json_converter(
        o
):

    if isinstance(
            o,
            np.integer
    ):

        return int(o)

    if isinstance(
            o,
            np.floating
    ):

        return float(o)

    if isinstance(
            o,
            np.ndarray
    ):

        return o.tolist()

    return str(o)

def sanitize_json(
        obj
):

    if isinstance(
            obj,
            dict
    ):

        return {

            str(
                sanitize_json(k)
            ):

            sanitize_json(v)

            for k,v in obj.items()
        }

    elif isinstance(
            obj,
            list
    ):

        return [

            sanitize_json(x)

            for x in obj
        ]

    elif isinstance(
            obj,
            tuple
    ):

        return tuple(

            sanitize_json(x)

            for x in obj
        )

    elif isinstance(
            obj,
            np.integer
    ):

        return int(obj)

    elif isinstance(
            obj,
            np.floating
    ):

        return float(obj)

    elif isinstance(
            obj,
            np.ndarray
    ):

        return obj.tolist()

    return obj

def save_csv(
        df,
        filename
):

    path = (
        RESULTS_DIR /
        filename
    )

    df.to_csv(
        path,
        index=False
    )

    print(
        f"Saved: {path}"
    )


def save_json(
        obj,
        filename
):

    path = (
        RESULTS_DIR /
        filename
    )

    with open(
            path,
            "w"
    ) as f:

        clean_obj = sanitize_json(
            obj
        )

        json.dump(

            clean_obj,
            f,

            indent=4

        )

    print(
        f"Saved: {path}"
    )