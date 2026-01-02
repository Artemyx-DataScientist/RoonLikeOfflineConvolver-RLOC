from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve

from smartroon.types import FilterConfig, FilterPath
from smartroon.zipio import read_bytes


def load_ir_from_zip(zip_path: Path | str, ir_inner_path: str) -> Tuple[np.ndarray, int]:
    """
    Загружает импульсную характеристику из ZIP.

    Args:
        zip_path: Путь к ZIP-архиву.
        ir_inner_path: Внутренний путь к файлу IR.

    Returns:
        Кортеж (ir, sample_rate).
    """

    data = read_bytes(zip_path, ir_inner_path)
    with BytesIO(data) as buffer:
        ir, sample_rate = sf.read(buffer, dtype="float64", always_2d=False)
    return np.asarray(ir, dtype=np.float64), int(sample_rate)


def _validate_audio_shape(audio: np.ndarray, expected_channels: int) -> np.ndarray:
    if audio.ndim == 1:
        audio = audio[:, np.newaxis]
    if audio.ndim != 2:
        raise ValueError("audio должен быть одномерным или двухмерным массивом")
    if audio.shape[1] != expected_channels:
        raise ValueError(
            f"ожидается {expected_channels} входных каналов, получено {audio.shape[1]}"
        )
    return audio


def _select_ir_channel(ir: np.ndarray, channel_index: int) -> np.ndarray:
    if ir.ndim == 1:
        if channel_index != 0:
            raise ValueError("моно IR поддерживает только канал 0")
        return ir
    if ir.ndim != 2:
        raise ValueError("IR должен быть одномерным или двумерным массивом")
    if channel_index < 0 or channel_index >= ir.shape[1]:
        raise ValueError(
            f"ir_channel={channel_index} вне диапазона для {ir.shape[1]} каналов"
        )
    return ir[:, channel_index]


def _validate_gains(path: FilterPath, num_in: int, num_out: int) -> None:
    if len(path.in_gains) != num_in:
        raise ValueError(
            f"для ir_path={path.ir_path} число in_gains={len(path.in_gains)} "
            f"не совпадает с num_in={num_in}"
        )
    if len(path.out_gains) != num_out:
        raise ValueError(
            f"для ir_path={path.ir_path} число out_gains={len(path.out_gains)} "
            f"не совпадает с num_out={num_out}"
        )


def convolve(
    audio: np.ndarray,
    sr: int,
    cfg: FilterConfig,
    zip_path: Path | str,
) -> np.ndarray:
    """
    Выполняет оффлайн-конволюцию по FilterConfig.

    Args:
        audio: Входной сигнал формы (N, cfg.num_in) или (N,).
        sr: Частота дискретизации входного сигнала.
        cfg: Конфигурация фильтров.
        zip_path: Путь к ZIP с IR.

    Returns:
        Выходной сигнал формы (N + max_ir_len - 1, cfg.num_out).
    """

    if sr != cfg.sample_rate:
        raise ValueError(
            f"sample_rate входа {sr} не совпадает с config {cfg.sample_rate}"
        )

    audio = np.asarray(audio, dtype=np.float64)
    audio = _validate_audio_shape(audio, cfg.num_in)

    paths_ir: List[Tuple[FilterPath, np.ndarray]] = []
    max_ir_len = 0
    for path in cfg.paths:
        _validate_gains(path, cfg.num_in, cfg.num_out)
        ir_data, ir_sr = load_ir_from_zip(zip_path, path.ir_path)
        if ir_sr != sr:
            raise ValueError(
                f"IR sample_rate {ir_sr} не совпадает с ожидаемым {sr} для {path.ir_path}"
            )
        h_channel = np.asarray(_select_ir_channel(ir_data, path.ir_channel), dtype=np.float64)
        max_ir_len = max(max_ir_len, h_channel.shape[0])
        paths_ir.append((path, h_channel))

    if max_ir_len == 0:
        raise ValueError("не удалось определить длину IR")

    output_length = audio.shape[0] + max_ir_len - 1
    output = np.zeros((output_length, cfg.num_out), dtype=np.float64)

    for path, h in paths_ir:
        x_path = np.zeros(audio.shape[0], dtype=np.float64)
        for idx, gain in enumerate(path.in_gains):
            x_path += audio[:, idx] * gain

        y_path = fftconvolve(x_path, h, mode="full")

        for out_idx, gain in enumerate(path.out_gains):
            output[: y_path.shape[0], out_idx] += y_path * gain

    return output
