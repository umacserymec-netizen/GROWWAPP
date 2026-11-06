"""Public configuration API.

Usage::

    from config import load_settings
    cfg = load_settings()

Never read ``.env`` or ``config.yaml`` anywhere else.
"""

from config.settings import (
    DATA_SUBDIRS,
    PROVIDER_ENV,
    ConfigError,
    Settings,
    ensure_data_dirs,
    load_settings,
)

__all__ = [
    "ConfigError",
    "DATA_SUBDIRS",
    "PROVIDER_ENV",
    "Settings",
    "ensure_data_dirs",
    "load_settings",
]
