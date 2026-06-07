# ######################################################################
# Author: Cameron M. Gattis
# Created: 2026-06-07
# Updated: New.
# Purpose: Provides a csv and JSON export capability for data.
# ######################################################################

import json
from pathlib import Path
import numpy as np
import pandas as pd


RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

def sanitize_json(obj):
    '''
        Provides JSON-safe conversion.

        @params obj : multitype

        @return obj : multitype
    '''
    if isinstance(obj,dict):
        return {
            str(sanitize_json(k)):
            sanitize_json(v)
            for k,v in obj.items()
        }
    elif isinstance(obj,list):
        return [
            sanitize_json(x)
            for x in obj
        ]
    elif isinstance(obj,tuple):
        return tuple(
            sanitize_json(x)
            for x in obj
        )
    elif isinstance(obj,np.integer):

        return int(obj)
    elif isinstance(obj,np.floating):
        return float(obj)

    elif isinstance(obj,np.ndarray):
        return obj.tolist()

    return obj

def save_csv(df,filename):
    '''
        Exports data to csv.

        @params df : dataframe
                filename : string
    '''
    path = (RESULTS_DIR / filename)
    df.to_csv(path,index=False)
    print(f"Saved: {path}")


def save_json(obj,filename):
    '''
        Exports data and cleans the data to JSON.

        @params df : object
                filename : string
    '''
    path = (RESULTS_DIR /filename)
    with open(path,"w") as f:
        clean_obj = sanitize_json(obj)
        json.dump(clean_obj,f,indent=4)
    print(f"Saved: {path}")