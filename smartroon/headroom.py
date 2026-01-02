from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import soundfile as sf

from smartroon.dsp.convolver import convolve
from smartroon.dsp.streaming_convolver import stream_convolve_to_file
from smartroon.dsp.truepeak import recommended_gain_db, true_peak_db
from smartroon.loaders import load_filter_from_zip
from smartroon.logging_utils import get_logger

logger = get_logger(__name__)
IN_MEMORY_ANALYSIS_FRAME_LIMIT = 10_000_000


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

    Примечание: анализ true peak выполняется в памяти, поэтому для часовых файлов и Hi-Res
    он может потребовать значительных ресурсов. Для стримингового расчёта потребуется
    отдельный двухпроходный алгоритм (следующее ТЗ).
    """

    if oversample <= 0:
        raise ValueError("oversample должен быть положительным")

    logger.info(
        "Запуск анализа headroom: audio=%s, filters=%s, target=%.2f dBFS, oversample=%d",
        audio_path,
        zip_path,
        target_db,
        oversample,
    )
    audio, sample_rate = load_audio(audio_path)
    logger.debug("Аудио загружено: sample_rate=%d, shape=%s", sample_rate, audio.shape)
    filter_configs = load_filter_from_zip(zip_path)
    logger.debug("Найдено конфигураций фильтра: %d", len(filter_configs))

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
    logger.info("True peak до обработки: %.4f dBFS", peak_db)
    logger.info("Рекомендуемый gain: %.4f dB (headroom %.4f dB)", gain_db, -gain_db)

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
    stream_only: bool = False,
    gain_db: float | None = None,
    chunk_size: int = 65_536,
    dtype: str = "float32",
    analysis_frame_limit: int = IN_MEMORY_ANALYSIS_FRAME_LIMIT,
) -> Dict[str, float | int | str]:
    """Выполняет стриминговую конволюцию и сохраняет результат с применённым gain.

    Args:
        zip_path: Путь к ZIP с фильтрами.
        audio_path: Путь к аудиофайлу.
        output_path: Путь для сохранения WAV (PCM_24).
        target_db: Целевой true peak в dBFS.
        oversample: Фактор оверсемплинга.
        stream_only: Пропустить анализ true peak и использовать переданный gain_db.
        gain_db: Предустановленный gain, если пользователь не хочет рассчитывать его.
        chunk_size: Размер блока выборок для стриминговой конволюции.
        dtype: Тип данных для внутренней обработки (например, ``\"float32\"``).
        analysis_frame_limit: Максимальное число фреймов для in-memory анализа.

    Returns:
        Словарь с отчётом: исходный true peak, рекомендуемый gain, итоговый true peak и путь к файлу.
    """

    if oversample <= 0:
        raise ValueError("oversample должен быть положительным")
    if chunk_size <= 0:
        raise ValueError("chunk_size должен быть положительным")

    logger.info(
        (
            "Запуск рендера: audio=%s, filters=%s, output=%s, target=%.2f dBFS, "
            "oversample=%d, stream_only=%s"
        ),
        audio_path,
        zip_path,
        output_path,
        target_db,
        oversample,
        stream_only,
    )

    filter_configs = load_filter_from_zip(zip_path)
    logger.debug("Найдено конфигураций фильтра: %d", len(filter_configs))

    with sf.SoundFile(audio_path, mode="r") as audio_file:
        sample_rate = int(audio_file.samplerate)
        frames = int(audio_file.frames)
        logger.debug(
            "Файл: samplerate=%d, channels=%d, frames=%d", sample_rate, audio_file.channels, frames
        )

    config = filter_configs.get(sample_rate)
    if config is None:
        raise ValueError(f"Не найден FilterConfig для sample_rate {sample_rate}")

    if config.num_in <= 0 or config.num_out <= 0:
        raise ValueError("FilterConfig содержит некорректные значения каналов")

    analysis_allowed = frames <= analysis_frame_limit and not stream_only

    audio_for_analysis: np.ndarray | None = None
    if analysis_allowed:
        audio_for_analysis, loaded_sr = load_audio(audio_path)
        if audio_for_analysis.shape[1] != config.num_in:
            raise ValueError(
                f"Число каналов аудио {audio_for_analysis.shape[1]} не совпадает с num_in={config.num_in}"
            )
        if loaded_sr != sample_rate:
            raise ValueError(
                f"sample_rate входа {loaded_sr} не совпадает с config {sample_rate}"
            )

    if not analysis_allowed and gain_db is None:
        raise ValueError(
            "Для больших файлов расчёт true peak в памяти отключён. "
            "Укажите --stream-only и передайте --gain-db, либо сократите audio."
        )

    peak_before_db: float | None = None
    peak_after_db: float | None = None
    applied_gain_db: float = gain_db if gain_db is not None else 0.0

    if analysis_allowed:
        convolved = convolve(audio_for_analysis, sample_rate, config, zip_path)  # type: ignore[arg-type]
        peak_before_db = true_peak_db(convolved, oversample=oversample)
        if gain_db is None:
            applied_gain_db = recommended_gain_db(peak_before_db, target_db=target_db)
        processed = apply_gain_db(convolved, applied_gain_db)
        peak_after_db = true_peak_db(processed, oversample=oversample)
        logger.info(
            "True peak: до=%.4f dBFS, после=%.4f dBFS, gain=%.4f dB",
            peak_before_db,
            peak_after_db,
            applied_gain_db,
        )
    else:
        logger.info("Пропускаем анализ true peak: stream_only=%s, frames=%d", stream_only, frames)

    stream_convolve_to_file(
        audio_path=audio_path,
        zip_path=zip_path,
        cfg=config,
        output_path=output_path,
        chunk_size=chunk_size,
        dtype=dtype,
        progress=True,
        gain_db=applied_gain_db,
    )
    logger.info("Результат сохранён: %s", output_path)

    return {
        "sample_rate": sample_rate,
        "true_peak_before_db": peak_before_db,
        "target_true_peak_db": float(target_db),
        "recommended_gain_db": applied_gain_db,
        "recommended_headroom_db": -applied_gain_db,
        "true_peak_after_db": peak_after_db,
        "oversample": oversample,
        "output_path": str(Path(output_path)),
    }


__all__ = ["analyze_headroom", "render_convolved", "apply_gain_db", "load_audio"]
