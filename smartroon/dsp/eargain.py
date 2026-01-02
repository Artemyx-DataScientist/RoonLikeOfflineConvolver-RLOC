from __future__ import annotations

from typing import Sequence

import numpy as np


def apply_ear_gain(audio: np.ndarray, gains_db: Sequence[float]) -> np.ndarray:
    """Применяет поканальный gain в dB к аудиосигналу.

    Args:
        audio: Аудиосигнал формы ``(N, C)`` или ``(N,)``.
        gains_db: Список значений gain в dB для каждого канала.

    Returns:
        Новый массив с применённым поканальным gain.
    """

    signal = np.asarray(audio, dtype=np.float64)
    if signal.ndim == 1:
        signal = signal[:, np.newaxis]
    if signal.ndim != 2:
        raise ValueError("audio должен быть одномерным или двумерным массивом")

    gains = np.asarray(list(gains_db), dtype=np.float64)
    if gains.shape[0] != signal.shape[1]:
        raise ValueError(
            f"длина gains_db ({gains.shape[0]}) не совпадает с числом каналов аудио ({signal.shape[1]})"
        )

    linear = 10.0 ** (gains / 20.0)
    return signal * linear[np.newaxis, :]


__all__ = ["apply_ear_gain"]
