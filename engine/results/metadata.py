from datetime import datetime

from engine.results.exporter import save_json


def save_run_metadata(
        filename="run_metadata.json",
        **kwargs
):

    metadata = {

        "timestamp":
            datetime.now().isoformat(),

        **kwargs

    }

    save_json(
        metadata,
        filename
    )