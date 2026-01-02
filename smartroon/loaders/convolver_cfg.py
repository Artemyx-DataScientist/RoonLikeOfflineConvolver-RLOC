from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Dict, List, Sequence

from smartroon.types import FilterConfig, FilterPath
from smartroon.zipio import list_files, read_text


def _normalize_zip_path(path: str | PurePosixPath) -> PurePosixPath:
    normalized = PurePosixPath(str(path))
    parts = tuple(part for part in normalized.parts if part not in ("", "."))
    return PurePosixPath(*parts)


def _collect_files(zip_path: Path | str) -> set[PurePosixPath]:
    return {_normalize_zip_path(name) for name in list_files(zip_path)}


def _parse_header(line: str) -> tuple[int, int, int]:
    parts = line.split()
    if len(parts) != 4:
        raise ValueError(f"Ожидалось 4 значения в заголовке cfg, получено: {line!r}")
    try:
        sample_rate, num_in, num_out = (int(parts[i]) for i in range(3))
    except ValueError as exc:
        raise ValueError(f"Не удалось разобрать заголовок cfg: {line!r}") from exc
    if sample_rate <= 0 or num_in <= 0 or num_out <= 0:
        raise ValueError(f"Значения заголовка должны быть положительными: {line!r}")
    return sample_rate, num_in, num_out


def _parse_float_list(line: str, expected_length: int, label: str) -> List[float]:
    parts = line.split()
    if len(parts) != expected_length:
        raise ValueError(
            f"Ожидалось {expected_length} значений для {label}, получено {len(parts)}: {line!r}"
        )
    try:
        return [float(value) for value in parts]
    except ValueError as exc:
        raise ValueError(f"Не удалось разобрать {label}: {line!r}") from exc


def _resolve_ir_path(config_path: PurePosixPath, filename: str) -> PurePosixPath:
    base = config_path.parent
    relative = PurePosixPath(filename)
    if str(base) in ("", "."):
        return _normalize_zip_path(relative)
    return _normalize_zip_path(base / relative)


def _calculate_delay(
    input_delays: Sequence[float],
    output_delays: Sequence[float],
    in_gains: Sequence[float],
    out_gains: Sequence[float],
) -> float:
    active_inputs = [index for index, gain in enumerate(in_gains) if gain != 0]
    active_outputs = [index for index, gain in enumerate(out_gains) if gain != 0]
    if len(active_inputs) == 1 and len(active_outputs) == 1:
        return float(input_delays[active_inputs[0]] + output_delays[active_outputs[0]])
    return 0.0


def find_cfg(zip_path: Path | str) -> List[str]:
    """Возвращает список путей .cfg внутри ZIP."""

    files = sorted(_collect_files(zip_path))
    return [str(path) for path in files if path.suffix.lower() == ".cfg"]


def load_cfg(zip_path: Path | str, cfg_inner_path: Path | str) -> FilterConfig:
    """Загружает convolver.cfg и возвращает FilterConfig."""

    config_path = _normalize_zip_path(PurePosixPath(cfg_inner_path))
    content = read_text(zip_path, config_path)
    raw_lines = [line.strip() for line in content.splitlines()]
    lines = [line for line in raw_lines if line and not line.startswith(("#", ";"))]

    if len(lines) < 3:
        raise ValueError(f"В cfg {config_path} недостаточно строк для заголовка и задержек")

    sample_rate, num_in, num_out = _parse_header(lines[0])
    input_delays = _parse_float_list(lines[1], num_in, "input delays")
    output_delays = _parse_float_list(lines[2], num_out, "output delays")

    zip_files = _collect_files(zip_path)
    paths: List[FilterPath] = []

    index = 3
    while index < len(lines):
        remaining = len(lines) - index
        if remaining < 4:
            raise ValueError(f"Неполный блок фильтра в {config_path} начиная с строки {index + 1}")

        ir_filename = lines[index]
        try:
            ir_channel = int(lines[index + 1])
        except ValueError as exc:
            raise ValueError(
                f"Индекс канала IR должен быть целым числом в {config_path}: {lines[index + 1]!r}"
            ) from exc
        if ir_channel < 0:
            raise ValueError(f"Индекс канала IR должен быть неотрицательным в {config_path}")

        in_gains = _parse_float_list(lines[index + 2], num_in, "in_gains")
        out_gains = _parse_float_list(lines[index + 3], num_out, "out_gains")
        ir_path = _resolve_ir_path(config_path, ir_filename)
        if ir_path not in zip_files:
            raise FileNotFoundError(
                f"IR-файл {ir_path} из cfg {config_path} не найден в архиве {zip_path}"
            )

        paths.append(
            FilterPath(
                in_gains=in_gains,
                out_gains=out_gains,
                ir_path=str(ir_path),
                ir_channel=ir_channel,
                delay_ms=_calculate_delay(input_delays, output_delays, in_gains, out_gains),
            )
        )
        index += 4

    return FilterConfig(sample_rate=sample_rate, num_in=num_in, num_out=num_out, paths=paths)


def load_all_cfg(zip_path: Path | str) -> Dict[int, FilterConfig]:
    """Загружает все .cfg из ZIP и объединяет FilterConfig по sample_rate."""

    configs: Dict[int, FilterConfig] = {}
    for config_location in find_cfg(zip_path):
        config = load_cfg(zip_path, config_location)
        existing = configs.get(config.sample_rate)
        if existing is None:
            configs[config.sample_rate] = config
            continue
        if existing.num_in != config.num_in or existing.num_out != config.num_out:
            raise ValueError(
                f"Конфликты num_in/num_out для sample_rate {config.sample_rate} между cfg файлами"
            )
        existing.paths.extend(config.paths)
    return configs
