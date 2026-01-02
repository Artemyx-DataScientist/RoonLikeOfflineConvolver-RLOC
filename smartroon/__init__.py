"""Утилиты SmartRoon для работы с офлайн-конвольвером."""

from importlib import metadata

from .types import FilterConfig, FilterPath
from .zipio import list_files, read_bytes, read_text

__all__ = [
    "FilterConfig",
    "FilterPath",
    "list_files",
    "read_bytes",
    "read_text",
]


def get_version() -> str:
    """Возвращает версию пакета, если он установлен."""

    try:
        return metadata.version("smartroon")
    except metadata.PackageNotFoundError:
        return "0.0.0"


__version__ = get_version()
