from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Sequence, Tuple

import numpy as np
import soundfile as sf
from scipy.signal import oaconvolve
from scipy.signal import resample_poly

from smartroon.types import FilterConfig, FilterPath

from .convolver import _select_ir_channel, _validate_gains, load_ir_from_zip


def _load_paths_ir(
    zip_path: Path | str, cfg: FilterConfig, sample_rate: int, dtype: np.dtype
) -> Tuple[List[Tuple[FilterPath, np.ndarray]], int]:
    paths_ir: List[Tuple[FilterPath, np.ndarray]] = []
    max_ir_len = 0

    for path in cfg.paths:
        _validate_gains(path, cfg.num_in, cfg.num_out)
        ir_data, ir_sr = load_ir_from_zip(zip_path, path.ir_path)
        if ir_sr != sample_rate:
            raise ValueError(
                f"IR sample_rate {ir_sr} не совпадает с ожидаемым {sample_rate} для {path.ir_path}"
            )
        h_channel = np.asarray(_select_ir_channel(ir_data, path.ir_channel), dtype=dtype)
        max_ir_len = max(max_ir_len, h_channel.shape[0])
        paths_ir.append((path, h_channel))

    if max_ir_len == 0:
        raise ValueError("не удалось определить длину IR")

    return paths_ir, max_ir_len


def _log_progress(
    processed_frames: int,
    total_frames: int,
    last_reported: int,
    logger_callback: Callable[[int], None],
) -> int:
    if total_frames <= 0:
        return last_reported

    percent = int(processed_frames * 100 / total_frames)
    if percent > last_reported:
        logger_callback(percent)
        return percent
    return last_reported


def _compute_true_peak_block(block: np.ndarray, oversample: int) -> float:
    oversampled = resample_poly(block, oversample, 1, axis=0, padtype="mean")
    return float(np.max(np.abs(oversampled)))


def _prepare_ear_gain_linear(
    ear_gains_db: Sequence[float] | None, num_channels: int, dtype: np.dtype
) -> np.ndarray | None:
    if ear_gains_db is None:
        return None
    if len(ear_gains_db) != num_channels:
        raise ValueError(
            f"ожидается {num_channels} значений ear gain, получено {len(ear_gains_db)}"
        )
    gains_db = np.asarray(ear_gains_db, dtype=np.float64)
    linear = (10.0 ** (gains_db / 20.0)).astype(dtype, copy=False)
    return linear


def stream_true_peak_db(
    audio_path: Path | str,
    zip_path: Path | str,
    cfg: FilterConfig,
    chunk_size: int = 65_536,
    oversample: int = 4,
    dtype: str = "float64",
    ear_gains_db: Sequence[float] | None = None,
) -> float:
    """Стримингово рассчитывает true peak конволюции без сохранения результата.

    Конволюция выполняется блоками, после чего каждый блок (с учётом перекрытия)
    оверсемплируется для оценки локального пика. В памяти одновременно находится
    только текущий блок и небольшой хвост для корректной обработки стыков.

    Args:
        audio_path: Путь к входному аудио.
        zip_path: Путь к ZIP с фильтрами.
        cfg: Конфигурация фильтра.
        chunk_size: Размер блока выборок.
        oversample: Фактор оверсемплинга.
        dtype: Тип данных для расчёта.
        ear_gains_db: Поканальные значения ear-gain в dB, применяемые к результату конволюции.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size должен быть положительным")
    if oversample <= 0:
        raise ValueError("oversample должен быть положительным")

    processing_dtype = np.dtype(dtype)
    ear_gain_linear = _prepare_ear_gain_linear(ear_gains_db, cfg.num_out, processing_dtype)
    oversample_margin = max(oversample * 10, 1)
    max_abs_value = 0.0

    with sf.SoundFile(audio_path, mode="r") as audio_file:
        sample_rate = int(audio_file.samplerate)
        if sample_rate != cfg.sample_rate:
            raise ValueError(
                f"sample_rate входа {sample_rate} не совпадает с config {cfg.sample_rate}"
            )

        if audio_file.channels != cfg.num_in:
            raise ValueError(
                f"Число каналов аудио {audio_file.channels} не совпадает с num_in={cfg.num_in}"
            )

        paths_ir, max_ir_len = _load_paths_ir(zip_path, cfg, sample_rate, processing_dtype)

        overlap = np.zeros((max_ir_len - 1, cfg.num_out), dtype=processing_dtype)
        prev_tail = np.zeros((0, cfg.num_out), dtype=processing_dtype)

        while True:
            chunk = audio_file.read(frames=chunk_size, dtype=processing_dtype, always_2d=True)
            if chunk.size == 0:
                break

            chunk_len = chunk.shape[0]
            result = np.zeros((chunk_len + max_ir_len - 1, cfg.num_out), dtype=processing_dtype)

            for path, h in paths_ir:
                x_path = np.zeros(chunk_len, dtype=processing_dtype)
                for idx, gain in enumerate(path.in_gains):
                    x_path += chunk[:, idx] * gain

                y_path = oaconvolve(x_path, h)

                for out_idx, gain in enumerate(path.out_gains):
                    result[: y_path.shape[0], out_idx] += y_path * gain

            if overlap.size:
                result[: overlap.shape[0]] += overlap

            if ear_gain_linear is not None:
                result *= ear_gain_linear

            block_output = result[:chunk_len]
            analysis_block = (
                np.concatenate([prev_tail, block_output], axis=0)
                if prev_tail.size
                else block_output
            )
            if analysis_block.size:
                max_abs_value = max(max_abs_value, _compute_true_peak_block(analysis_block, oversample))

            prev_tail = block_output[-oversample_margin:].copy()
            overlap = result[chunk_len:]

        if overlap.size:
            analysis_block = (
                np.concatenate([prev_tail, overlap], axis=0) if prev_tail.size else overlap
            )
            if analysis_block.size:
                max_abs_value = max(max_abs_value, _compute_true_peak_block(analysis_block, oversample))

    eps = float(np.finfo(np.float64).eps)
    return float(20.0 * np.log10(max_abs_value + eps))


def stream_convolve_to_file(
    audio_path: Path | str,
    zip_path: Path | str,
    cfg: FilterConfig,
    output_path: Path | str,
    chunk_size: int = 65_536,
    dtype: str = "float32",
    progress: bool = True,
    gain_db: float = 0.0,
    output_subtype: str = "PCM_24",
    ear_gains_db: Sequence[float] | None = None,
) -> None:
    """Выполняет стриминговую конволюцию и сохраняет результат в WAV.

    Конволюция выполняется блоками (чанками) с overlap-add. В памяти находятся
    только текущий блок входа, хвост перекрытия и загруженные IR.

    Args:
        audio_path: Путь к входному аудио.
        zip_path: Путь к ZIP с фильтрами.
        cfg: Конфигурация фильтров.
        output_path: Путь для сохранения результата.
        chunk_size: Размер блока выборок.
        dtype: Тип данных для расчёта.
        progress: Выводить прогресс.
        gain_db: Глобальный gain для результата после ear-gain.
        output_subtype: Подтип WAV.
        ear_gains_db: Поканальный ear-gain в dB для результата конволюции.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size должен быть положительным")

    processing_dtype = np.dtype(dtype)
    ear_gain_linear = _prepare_ear_gain_linear(ear_gains_db, cfg.num_out, processing_dtype)
    logger_callback: Callable[[int], None] | None = None
    if progress:
        from smartroon.logging_utils import get_logger

        progress_logger = get_logger(__name__)

        def _report_progress(percent: int) -> None:
            progress_logger.info("Прогресс: %d%%", percent)

        logger_callback = _report_progress

    gain_linear = processing_dtype.type(10.0 ** (gain_db / 20.0))

    with sf.SoundFile(audio_path, mode="r") as audio_file:
        sample_rate = int(audio_file.samplerate)
        if sample_rate != cfg.sample_rate:
            raise ValueError(
                f"sample_rate входа {sample_rate} не совпадает с config {cfg.sample_rate}"
            )

        if audio_file.channels != cfg.num_in:
            raise ValueError(
                f"Число каналов аудио {audio_file.channels} не совпадает с num_in={cfg.num_in}"
            )

        paths_ir, max_ir_len = _load_paths_ir(zip_path, cfg, sample_rate, processing_dtype)

        output_target = Path(output_path)
        output_target.parent.mkdir(parents=True, exist_ok=True)
        overlap = np.zeros((max_ir_len - 1, cfg.num_out), dtype=processing_dtype)
        last_reported_percent = -1

        with sf.SoundFile(
            output_target,
            mode="w",
            samplerate=sample_rate,
            channels=cfg.num_out,
            format="WAV",
            subtype=output_subtype,
        ) as output_file:
            while True:
                chunk = audio_file.read(frames=chunk_size, dtype=processing_dtype, always_2d=True)
                if chunk.size == 0:
                    break

                chunk_len = chunk.shape[0]
                result = np.zeros((chunk_len + max_ir_len - 1, cfg.num_out), dtype=processing_dtype)

                for path, h in paths_ir:
                    x_path = np.zeros(chunk_len, dtype=processing_dtype)
                    for idx, gain in enumerate(path.in_gains):
                        x_path += chunk[:, idx] * gain

                    y_path = oaconvolve(x_path, h)

                    for out_idx, gain in enumerate(path.out_gains):
                        result[: y_path.shape[0], out_idx] += y_path * gain

                if overlap.size:
                    result[: overlap.shape[0]] += overlap

                if ear_gain_linear is not None:
                    result *= ear_gain_linear

                if gain_linear != 1:
                    result *= gain_linear

                output_file.write(result[:chunk_len])
                overlap = result[chunk_len:]

                if progress and logger_callback is not None:
                    last_reported_percent = _log_progress(
                        processed_frames=audio_file.tell(),
                        total_frames=audio_file.frames,
                        last_reported=last_reported_percent,
                        logger_callback=logger_callback,
                    )

            if overlap.size:
                output_file.write(overlap)
