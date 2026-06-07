# ######################################################################
# Author: Cameron M. Gattis
# Created: 2026-06-07
# Updated: New.
# Purpose: Provides a JSON export capability for metadata.
# ######################################################################

from datetime import datetime
from engine.results.exporter import save_json


def save_run_metadata(filename="run_metadata.json",**kwargs):
    '''
        Purpose of function/method.

        @params timestamp : datetime object
                kwargs    : misc data
    '''
    metadata = {
        "timestamp": datetime.now().isoformat(),
        **kwargs
    }

    save_json(metadata,filename)