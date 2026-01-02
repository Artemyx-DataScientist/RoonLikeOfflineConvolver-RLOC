from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import soundfile as sf

from smartroon.dsp.convolver import convolve
from smartroon.dsp.truepeak import recommended_gain_db, true_peak_db
from smartroon.loaders import load_filter_from_zip


def load_audio(audio_path: Path | str) -> Tuple[np.ndarray, int]:
    """Загружает аудиофайл и возвращает данные в форме ``(N, C)``.

    Args:
        audio_path: Путь к аудиофайлу.

    Returns:
        Кортеж ``(audio, sample_rate)``.
    """

    path = Path(audio_path)
    data, sample_rate = sf.read(path, dtype="float64", always_2d=False)
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

    if oversample <= 0:
        raise ValueError("oversample должен быть положительным")

    audio, sample_rate = load_audio(audio_path)
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


def apply_gain_db(audio: np.ndarray, gain_db: float) -> np.ndarray:
    """Возвращает сигнал с применённым gain в дБ.

    Args:
        audio: Входной сигнал формы ``(N, C)`` или ``(N,)``.
        gain_db: Значение усиления/ослабления в дБ.

    Returns:
        Массива с применённым gain.
    """

    gain = float(10.0 ** (gain_db / 20.0))
    return np.asarray(audio, dtype=np.float64) * gain


def render_convolved(
    zip_path: Path | str,
    audio_path: Path | str,
    output_path: Path | str,
    target_db: float = -0.1,
    oversample: int = 4,
) -> Dict[str, float | int | str]:
    """Выполняет конволюцию и сохраняет результат с применённым gain.

    Args:
        zip_path: Путь к ZIP с фильтрами.
        audio_path: Путь к аудиофайлу.
        output_path: Путь для сохранения WAV (PCM_24).
        target_db: Целевой true peak в dBFS.
        oversample: Фактор оверсемплинга.

    Returns:
        Словарь с отчётом: исходный true peak, рекомендуемый gain, итоговый true peak и путь к файлу.
    """

    if oversample <= 0:
        raise ValueError("oversample должен быть положительным")

    audio, sample_rate = load_audio(audio_path)
    filter_configs = load_filter_from_zip(zip_path)

    config = filter_configs.get(sample_rate)
    if config is None:
        raise ValueError(f"Не найден FilterConfig для sample_rate {sample_rate}")

    if audio.shape[1] != config.num_in:
        raise ValueError(
            f"Число каналов аудио {audio.shape[1]} не совпадает с num_in={config.num_in}"
        )

    convolved = convolve(audio, sample_rate, config, zip_path)
    peak_before_db = true_peak_db(convolved, oversample=oversample)
    gain_db = recommended_gain_db(peak_before_db, target_db=target_db)
    processed = apply_gain_db(convolved, gain_db)
    peak_after_db = true_peak_db(processed, oversample=oversample)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output, processed, sample_rate, format="WAV", subtype="PCM_24")

    return {
        "sample_rate": sample_rate,
        "true_peak_before_db": peak_before_db,
        "target_true_peak_db": float(target_db),
        "recommended_gain_db": gain_db,
        "recommended_headroom_db": -gain_db,
        "true_peak_after_db": peak_after_db,
        "oversample": oversample,
        "output_path": str(output),
    }


__all__ = ["analyze_headroom", "render_convolved", "apply_gain_db", "load_audio"]
