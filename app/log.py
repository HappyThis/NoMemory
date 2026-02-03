from __future__ import annotations

import logging


def get_logger(name: str) -> logging.Logger:
    """
    Unified logger entrypoint.

    All runtime formatting/handlers/levels are configured centrally in `app.logging_setup.setup_logging()`.
    """

    return logging.getLogger(name)

