from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Dict, List

from smartroon.types import FilterConfig, FilterPath
from smartroon.zipio import list_files, read_text


def _normalize_zip_path(path: str | PurePosixPath) -> PurePosixPath:
    """Нормализует путь внутри ZIP к POSIX-виду без точек."""

    normalized = PurePosixPath(str(path))
    parts = tuple(part for part in normalized.parts if part not in (".", ""))
    return PurePosixPath(*parts)


def _collect_files(zip_path: Path | str) -> set[PurePosixPath]:
    return {_normalize_zip_path(name) for name in list_files(zip_path)}


def _resolve_ir_path(config_path: PurePosixPath, filename: str) -> PurePosixPath:
    relative = PurePosixPath(filename)
    base = config_path.parent
    if str(base) in ("", "."):
        return _normalize_zip_path(relative)
    return _normalize_zip_path(base / relative)


def _parse_header(line: str) -> tuple[int, int, int]:
    parts = line.split()
    if len(parts) != 3:
        raise ValueError(f"Ожидалось 3 числа в заголовке блока, получено: {line!r}")
    try:
        sr, num_in, num_out = (int(value) for value in parts)
    except ValueError as exc:
        raise ValueError(f"Не удалось прочитать заголовок блока: {line!r}") from exc
    if sr <= 0 or num_in <= 0 or num_out <= 0:
        raise ValueError(f"Значения заголовка должны быть положительными: {line!r}")
    return sr, num_in, num_out


def _parse_path_line(
    line: str,
    num_in: int,
    num_out: int,
    config_path: PurePosixPath,
    zip_files: set[PurePosixPath],
) -> FilterPath:
    parts = line.split()
    if len(parts) != 5:
        raise ValueError(f"Ожидалось 5 элементов в строке пути, получено: {line!r}")

    try:
        in_ch = int(parts[0])
        out_ch = int(parts[1])
        delay_ms = float(parts[2])
        gain_db = float(parts[3])
    except ValueError as exc:
        raise ValueError(f"Не удалось разобрать числовые значения строки: {line!r}") from exc

    if not 0 <= in_ch < num_in:
        raise ValueError(f"in_ch {in_ch} вне диапазона [0, {num_in})")
    if not 0 <= out_ch < num_out:
        raise ValueError(f"out_ch {out_ch} вне диапазона [0, {num_out})")

    ir_path = _resolve_ir_path(config_path, parts[4])
    if ir_path not in zip_files:
        raise FileNotFoundError(f"IR-файл {ir_path} не найден в ZIP")

    in_gains = [0.0 for _ in range(num_in)]
    out_gains = [0.0 for _ in range(num_out)]
    in_gains[in_ch] = 1.0
    out_gains[out_ch] = 10 ** (gain_db / 20)

    return FilterPath(
        in_gains=in_gains,
        out_gains=out_gains,
        ir_path=str(ir_path),
        ir_channel=0,
        delay_ms=delay_ms,
    )


def find_kemar_config(zip_path: Path | str) -> str:
    """Находит путь к config.txt внутри ZIP."""

    files = sorted(_collect_files(zip_path))
    for file_path in files:
        if file_path.name.lower() == "config.txt":
            return str(file_path)
    raise FileNotFoundError(f"config.txt не найден в архиве {zip_path}")


def load_kemar(zip_path: Path | str) -> Dict[int, FilterConfig]:
    """Загружает KEMAR/Roon config.txt и возвращает FilterConfig по sample_rate."""

    config_location = PurePosixPath(find_kemar_config(zip_path))
    zip_files = _collect_files(zip_path)
    content = read_text(zip_path, config_location)

    lines: List[str] = [line.strip() for line in content.splitlines() if line.strip()]
    configs: Dict[int, FilterConfig] = {}
    index = 0

    while index < len(lines):
        sr, num_in, num_out = _parse_header(lines[index])
        if sr in configs:
            raise ValueError(f"Дубликат блока для sample_rate {sr}")

        index += 1
        paths: List[FilterPath] = []

        while index < len(lines):
            parts = lines[index].split()
            if len(parts) == 3:
                # начало следующего блока
                break
            paths.append(_parse_path_line(lines[index], num_in, num_out, config_location, zip_files))
            index += 1

        configs[sr] = FilterConfig(sample_rate=sr, num_in=num_in, num_out=num_out, paths=paths)

    return configs
