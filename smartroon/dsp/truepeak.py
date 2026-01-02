from __future__ import annotations

import numpy as np
from scipy.signal import resample_poly


def true_peak_db(audio: np.ndarray, oversample: int = 4) -> float:
    """Вычисляет true peak в dBFS с учётом оверсемплинга.

    Args:
        audio: Аудиосигнал формы ``(N,)`` или ``(N, C)``.
        oversample: Фактор оверсемплинга. Должен быть положительным.

    Returns:
        Значение пика в dBFS.
    """

    if oversample <= 0:
        raise ValueError("oversample должен быть положительным")

    signal = np.asarray(audio, dtype=np.float64)
    if signal.ndim == 1:
        signal = signal[:, np.newaxis]
    if signal.ndim != 2:
        raise ValueError("audio должен быть одномерным или двумерным массивом")

    upsampled = resample_poly(signal, oversample, 1, axis=0, padtype="mean")
    max_abs = float(np.max(np.abs(upsampled)))
    eps = float(np.finfo(np.float64).eps)
    return float(20.0 * np.log10(max_abs + eps))


def recommended_gain_db(peak_db: float, target_db: float = -0.1, allow_boost: bool = False) -> float:
    """Возвращает рекомендуемое усиление/ослабление, чтобы достичь целевого пика.

    Args:
        peak_db: Текущий true peak в dBFS.
        target_db: Желаемый максимум true peak в dBFS.
        allow_boost: Разрешить положительный gain, если сигнал тише цели.

    Returns:
        Рекомендуемое значение gain в dB. Отрицательное — ослабление, положительное — усиление.
    """

    if peak_db <= target_db:
        return float(target_db - peak_db if allow_boost else 0.0)
    return float(target_db - peak_db)


__all__ = ["true_peak_db", "recommended_gain_db"]
