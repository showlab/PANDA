"""Static PANDA dataset protocol configuration.

Provider resolution lives in :mod:`panda.providers`. Importing this module may
read an explicitly selected local ``.env`` file, but never constructs clients,
opens network connections, or creates files.
"""

import os
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_environment() -> None:
    """Load the selected local environment file without overriding exported values."""
    try:
        from dotenv import load_dotenv

        load_dotenv(
            dotenv_path=os.getenv("PANDA_ENV_FILE", str(PROJECT_ROOT / ".env")),
            override=False,
        )
    except ImportError:
        pass


load_environment()

DATASET_CONFIGS: Dict[str, Dict[str, Any]] = {
    "ucf": {
        "dataset_name": "ucf",
        "primary_metric": "auc",
        "query_template": (
            "Please help me detect if any of the following abnormal events have "
            "occurred at the moment: Abuse, Arrest, Arson, Assault, Burglary, "
            "Explosion, Fighting, Road Accidents, Robbery, Shooting, Shoplifting, "
            "Stealing, Vandalism."
        ),
        "window": 5,
        "fps": 1,
        "num_rules": 5,
    },
    "xd": {
        "dataset_name": "xd",
        "primary_metric": "ap",
        "query_template": "Please help me detect if any violent event has occurred at the moment.",
        "window": 5,
        "fps": 1,
        "num_rules": 5,
    },
    "ub": {
        "dataset_name": "ub",
        "primary_metric": "auc",
        "query_template": "Please help me detect if any abnormal event has occurred at the moment.",
        "window": 5,
        "fps": 1,
        "num_rules": 5,
    },
    "csad": {
        "dataset_name": "csad",
        "primary_metric": "auc",
        "query_template": "Please help me detect if any abnormal event has occurred at the moment.",
        "window": 5,
        "fps": 1,
        "num_rules": 5,
    },
}
