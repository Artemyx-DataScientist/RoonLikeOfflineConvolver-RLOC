from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import soundfile as sf

from smartroon.dsp.convolver import convolve
from smartroon.dsp.truepeak import recommended_gain_db, true_peak_db
from smartroon.loaders import load_filter_from_zip


def _load_audio(audio_path: Path | str) -> tuple[np.ndarray, int]:
    data, sample_rate = sf.read(audio_path, dtype="float64", always_2d=False)
    audio = np.asarray(data, dtype=np.float64)
    if audio.ndim == 1:
        audio = audio[:, np.newaxis]
    if audio.ndim != 2:
        raise ValueError("audio должен быть одномерным или двухмерным массивом")
    return audio, int(sample_rate)


def analyze_headroom(
    zip_path: Path | str, audio_path: Path | str, target_db: float = -0.1, oversample: int = 4
) -> Dict[str, float | int]:
    """Анализирует true peak и рекомендуемый headroom для обработанного сигнала.

    Args:
        zip_path: Путь к ZIP-файлу с фильтрами.
        audio_path: Путь к аудиофайлу для анализа.
        target_db: Целевой максимум true peak в dBFS.
        oversample: Фактор оверсемплинга для измерения true peak.

    Returns:
        Словарь с полями ``sample_rate``, ``true_peak_before_db``, ``target_true_peak_db``,
        ``recommended_gain_db`` и ``recommended_headroom_db``.
    """

    audio, sample_rate = _load_audio(audio_path)
    filter_configs = load_filter_from_zip(zip_path)

    config = filter_configs.get(sample_rate)
    if config is None:
        raise ValueError(f"Не найден FilterConfig для sample_rate {sample_rate}")

    if audio.shape[1] != config.num_in:
        raise ValueError(
            f"Число каналов аудио {audio.shape[1]} не совпадает с num_in={config.num_in}"
        )

    convolved = convolve(audio, sample_rate, config, zip_path)
    peak_db = true_peak_db(convolved, oversample=oversample)
    gain_db = recommended_gain_db(peak_db, target_db=target_db)

    return {
        "sample_rate": sample_rate,
        "true_peak_before_db": peak_db,
        "target_true_peak_db": float(target_db),
        "recommended_gain_db": gain_db,
        "recommended_headroom_db": -gain_db,
    }


__all__ = ["analyze_headroom"]
