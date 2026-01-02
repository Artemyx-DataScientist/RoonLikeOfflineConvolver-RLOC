from __future__ import annotations

from pathlib import Path
from typing import Dict

from smartroon.types import FilterConfig

from .convolver_cfg import find_cfg, load_all_cfg, load_cfg
from .kemar import find_kemar_config, load_kemar


def load_filter_from_zip(zip_path: Path | str) -> Dict[int, FilterConfig]:
    """Определяет формат конфига и загружает фильтры.

    Предпочтительно используются файлы ``.cfg`` (Convolver). Если они отсутствуют,
    пробуем формат KEMAR (``config.txt``). Если ни один формат не найден, выбрасывается
    информативное исключение.
    """

    path = Path(zip_path)

    cfg_files = find_cfg(path)
    if cfg_files:
        return load_all_cfg(path)

    try:
        find_kemar_config(path)
    except FileNotFoundError:
        raise ValueError(
            f"В архиве {path} не найден поддерживаемый конфиг (.cfg или config.txt)"
        ) from None

    return load_kemar(path)


__all__ = [
    "find_kemar_config",
    "load_kemar",
    "find_cfg",
    "load_cfg",
    "load_all_cfg",
    "load_filter_from_zip",
]
