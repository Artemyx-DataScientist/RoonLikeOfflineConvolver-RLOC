from __future__ import annotations

import logging
import os
import sys
from typing import Optional

_LOGGER_NAME = "smartroon"
_LOG_FORMAT = "%(levelname)s [%(name)s] %(message)s"
_configured = False


def _determine_level() -> int:
    env_value = os.getenv("SMARTROON_DEBUG", "").strip()
    if env_value == "1":
        return logging.DEBUG
    return logging.INFO


def _configure_root(default_level: int) -> None:
    global _configured
    if _configured:
        return
    root_logger = logging.getLogger()
    has_stdout_handler = any(
        isinstance(handler, logging.StreamHandler)
        and getattr(handler, "_smartroon_managed", False)
        for handler in root_logger.handlers
    )
    if not root_logger.handlers or not has_stdout_handler:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        handler._smartroon_managed = True  # type: ignore[attr-defined]
        root_logger.addHandler(handler)
    root_logger.setLevel(default_level)
    _configured = True


def _refresh_stream_handlers() -> None:
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if isinstance(handler, logging.StreamHandler) and getattr(handler, "_smartroon_managed", False):
            handler.setStream(sys.stdout)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Возвращает сконфигурированный логгер SmartRoon.

    Root-логгер настраивается один раз (Singleton init). По умолчанию уровень INFO,
    при ``SMARTROON_DEBUG=1`` — DEBUG.
    """

    default_level = _determine_level()
    _configure_root(default_level)
    _refresh_stream_handlers()
    return logging.getLogger(name or _LOGGER_NAME)


__all__ = ["get_logger"]
