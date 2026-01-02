from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np
import soundfile as sf
from scipy.signal import oaconvolve

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
) -> None:
    """Выполняет стриминговую конволюцию и сохраняет результат в WAV.

    Конволюция выполняется блоками (чанками) с overlap-add. В памяти находятся
    только текущий блок входа, хвост перекрытия и загруженные IR.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size должен быть положительным")

    processing_dtype = np.dtype(dtype)
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
