"""Утилиты SmartRoon для работы с офлайн-конвольвером."""

import importlib.metadata as importlib_metadata

from .types import FilterConfig, FilterPath
from .zipio import list_files, read_bytes, read_text
from .loaders import find_kemar_config, load_filter_from_zip, load_kemar
from .dsp import convolve, load_ir_from_zip
from .metadata import copy_metadata

__all__ = [
    "FilterConfig",
    "FilterPath",
    "find_kemar_config",
    "load_kemar",
    "load_filter_from_zip",
    "convolve",
    "load_ir_from_zip",
    "copy_metadata",
    "list_files",
    "read_bytes",
    "read_text",
]


def get_version() -> str:
    """Возвращает версию пакета, если он установлен."""

    try:
        return importlib_metadata.version("smartroon")
    except importlib_metadata.PackageNotFoundError:
        return "0.0.0"


__version__ = get_version()
