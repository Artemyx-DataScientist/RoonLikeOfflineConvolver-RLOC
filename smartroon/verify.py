from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import soundfile as sf

from smartroon.dsp.convolver import convolve, load_ir_from_zip
from smartroon.dsp.truepeak import true_peak_db
from smartroon.headroom import load_audio
from smartroon.loaders import load_filter_from_zip
from smartroon.types import FilterConfig

DEFAULT_SECONDS = 5.0
DEFAULT_CHECKSUM_SAMPLES = 1_000_000
DEFAULT_OVERSAMPLE = 4


@dataclass(frozen=True, slots=True)
class VerificationMetrics:
    rms_per_channel: List[float]
    sample_peak: float
    true_peak: float
    checksum: str
    checksum_samples_used: int


@dataclass(frozen=True, slots=True)
class VerificationResult:
    report: Dict[str, Any]
    formatted: str
    snippet_in_path: Path
    snippet_out_path: Path
    report_path: Path


def _calculate_rms_per_channel(audio: np.ndarray) -> List[float]:
    data = np.asarray(audio, dtype=np.float64)
    if data.ndim == 1:
        data = data[:, np.newaxis]
    if data.size == 0:
        return []
    return [float(np.sqrt(np.mean(data[:, idx] ** 2))) for idx in range(data.shape[1])]


def _calculate_checksum(audio: np.ndarray, max_samples: int) -> Tuple[str, int]:
    flattened = np.ravel(np.asarray(audio, dtype=np.float64)).astype(np.float32)
    used = min(flattened.shape[0], max_samples)
    digest = hashlib.sha256(flattened[:used].tobytes()).hexdigest()
    return digest, used


def _collect_metrics(audio: np.ndarray, checksum_samples: int, oversample: int) -> VerificationMetrics:
    rms_values = _calculate_rms_per_channel(audio)
    sample_peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    true_peak_value = true_peak_db(audio, oversample=oversample)
    checksum, used_samples = _calculate_checksum(audio, checksum_samples)
    return VerificationMetrics(
        rms_per_channel=rms_values,
        sample_peak=sample_peak,
        true_peak=true_peak_value,
        checksum=checksum,
        checksum_samples_used=used_samples,
    )


def _select_config(filter_configs: Dict[int, FilterConfig], sample_rate: int) -> FilterConfig:
    config = filter_configs.get(sample_rate)
    if config is None:
        raise ValueError(f"Не найден FilterConfig для sample_rate {sample_rate}")
    return config


def _prepare_output_dir(audio_path: Path, output_dir: Path | None) -> Path:
    if output_dir is not None:
        base_dir = output_dir
    else:
        base_dir = audio_path.parent / f"{audio_path.stem}_verify"
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


def _slice_audio(audio: np.ndarray, sample_rate: int, seconds: float) -> np.ndarray:
    frames = min(audio.shape[0], int(sample_rate * seconds))
    if frames <= 0:
        raise ValueError("Фрагмент аудио пуст — проверьте значение seconds или длину файла")
    return np.asarray(audio[:frames, :], dtype=np.float64)


def _ir_lengths(config: FilterConfig, zip_path: Path | str) -> List[int]:
    lengths: List[int] = []
    for path in config.paths:
        ir_data, ir_sr = load_ir_from_zip(zip_path, path.ir_path)
        if ir_sr != config.sample_rate:
            raise ValueError(
                f"IR sample_rate {ir_sr} не совпадает с config {config.sample_rate} для {path.ir_path}"
            )
        lengths.append(int(ir_data.shape[0]))
    return lengths


def _format_metrics(label: str, metrics: VerificationMetrics) -> Iterable[str]:
    rms_values = ", ".join(f"{value:.6f}" for value in metrics.rms_per_channel)
    return [
        f"[{label}] RMS per channel: {rms_values}",
        f"[{label}] Sample peak: {metrics.sample_peak:.6f}",
        f"[{label}] True peak: {metrics.true_peak:.6f} dBFS",
        f"[{label}] SHA256: {metrics.checksum} (samples={metrics.checksum_samples_used})",
    ]


def _write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    sf.write(path, np.asarray(audio, dtype=np.float64), sample_rate, format="WAV", subtype="DOUBLE")


def _write_report(path: Path, report: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)


def run_verify(
    audio_path: Path | str,
    zip_path: Path | str,
    seconds: float = DEFAULT_SECONDS,
    output_dir: Path | None = None,
    checksum_samples: int = DEFAULT_CHECKSUM_SAMPLES,
    oversample: int = DEFAULT_OVERSAMPLE,
) -> VerificationResult:
    """
    Выполняет проверку конволюции на фрагменте аудио и сохраняет артефакты.

    Args:
        audio_path: Путь к входному аудиофайлу.
        zip_path: Путь к ZIP с фильтрами.
        seconds: Длительность фрагмента для проверки.
        output_dir: Каталог для сохранения результатов. Если не указан, используется ``<audio>_verify``.
        checksum_samples: Число сэмплов для расчёта SHA256 (flattened, float32).
        oversample: Фактор оверсемплинга для true peak.

    Returns:
        VerificationResult с путями к артефактам и словарём отчёта.
    """

    if seconds <= 0:
        raise ValueError("seconds должен быть положительным")
    if checksum_samples <= 0:
        raise ValueError("checksum_samples должен быть положительным")
    if oversample <= 0:
        raise ValueError("oversample должен быть положительным")

    audio_file = Path(audio_path)
    audio, sample_rate = load_audio(audio_file)

    filter_configs = load_filter_from_zip(zip_path)
    config = _select_config(filter_configs, sample_rate)
    snippet_in = _slice_audio(audio, sample_rate, seconds)
    ir_lengths = _ir_lengths(config, zip_path)

    convolved = convolve(snippet_in, sample_rate, config, zip_path)

    input_metrics = _collect_metrics(snippet_in, checksum_samples, oversample)
    output_metrics = _collect_metrics(convolved, checksum_samples, oversample)

    base_dir = _prepare_output_dir(audio_file, output_dir)
    snippet_in_path = base_dir / "snippet_in.wav"
    snippet_out_path = base_dir / "snippet_out.wav"
    report_path = base_dir / "report.json"

    _write_wav(snippet_in_path, snippet_in, sample_rate)
    _write_wav(snippet_out_path, convolved, sample_rate)

    report: Dict[str, Any] = {
        "sample_rate": sample_rate,
        "seconds": float(seconds),
        "input_frames": int(snippet_in.shape[0]),
        "output_frames": int(convolved.shape[0]),
        "ir_lengths": ir_lengths,
        "checksum_samples": checksum_samples,
        "paths_count": len(config.paths),
        "input": {
            "rms": input_metrics.rms_per_channel,
            "sample_peak": input_metrics.sample_peak,
            "true_peak_db": input_metrics.true_peak,
            "checksum": input_metrics.checksum,
            "checksum_samples_used": input_metrics.checksum_samples_used,
        },
        "output": {
            "rms": output_metrics.rms_per_channel,
            "sample_peak": output_metrics.sample_peak,
            "true_peak_db": output_metrics.true_peak,
            "checksum": output_metrics.checksum,
            "checksum_samples_used": output_metrics.checksum_samples_used,
        },
        "files": {
            "snippet_in": str(snippet_in_path),
            "snippet_out": str(snippet_out_path),
            "report": str(report_path),
        },
    }
    _write_report(report_path, report)

    lines: List[str] = [
        f"Sample rate: {sample_rate}",
        f"Frames analyzed: {snippet_in.shape[0]} (seconds={seconds})",
        f"IR lengths: {', '.join(str(length) for length in ir_lengths)}",
    ]
    lines.extend(_format_metrics("Input", input_metrics))
    lines.extend(_format_metrics("Output", output_metrics))
    formatted = "\n".join(lines)

    return VerificationResult(
        report=report,
        formatted=formatted,
        snippet_in_path=snippet_in_path,
        snippet_out_path=snippet_out_path,
        report_path=report_path,
    )


__all__ = ["run_verify", "VerificationResult", "VerificationMetrics"]
