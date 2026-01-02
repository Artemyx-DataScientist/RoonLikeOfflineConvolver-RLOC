from __future__ import annotations

from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import soundfile as sf

from smartroon.dsp.convolver import convolve
from smartroon.types import FilterConfig, FilterPath


def _wav_bytes(data: np.ndarray, sample_rate: int) -> bytes:
    buffer = BytesIO()
    sf.write(buffer, data, sample_rate, format="WAV", subtype="DOUBLE")
    return buffer.getvalue()


def test_identity_convolution(tmp_path: Path) -> None:
    sample_rate = 48_000
    ir_path = "ir/identity.wav"

    audio = np.array([0.1, -0.2, 0.3, 0.0], dtype=np.float64)
    config = FilterConfig(
        sample_rate=sample_rate,
        num_in=1,
        num_out=1,
        paths=[
            FilterPath(
                in_gains=[1.0],
                out_gains=[1.0],
                ir_path=ir_path,
                ir_channel=0,
            )
        ],
    )

    archive_path = tmp_path / "identity.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr(ir_path, _wav_bytes(np.array([1.0], dtype=np.float64), sample_rate))

    result = convolve(audio, sample_rate, config, archive_path)

    assert result.shape == (audio.shape[0], 1)
    assert np.allclose(result[:, 0], audio)


def test_stereo_crossfeed_toy(tmp_path: Path) -> None:
    sample_rate = 48_000
    ir_path = "ir/impulse.wav"

    audio_left = np.array([0.5, 0.0, -0.5, 1.0], dtype=np.float64)
    audio_right = np.array([1.0, -1.0, 0.5, 0.0], dtype=np.float64)
    audio = np.stack([audio_left, audio_right], axis=1)

    paths = [
        FilterPath(in_gains=[1.0, 0.0], out_gains=[1.0, 0.0], ir_path=ir_path, ir_channel=0),
        FilterPath(in_gains=[0.0, 1.0], out_gains=[0.0, 1.0], ir_path=ir_path, ir_channel=0),
        FilterPath(in_gains=[1.0, 0.0], out_gains=[0.0, 0.5], ir_path=ir_path, ir_channel=0),
        FilterPath(in_gains=[0.0, 1.0], out_gains=[0.5, 0.0], ir_path=ir_path, ir_channel=0),
    ]
    config = FilterConfig(sample_rate=sample_rate, num_in=2, num_out=2, paths=paths)

    archive_path = tmp_path / "crossfeed.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr(ir_path, _wav_bytes(np.array([1.0], dtype=np.float64), sample_rate))

    result = convolve(audio, sample_rate, config, archive_path)

    expected_left = audio_left + 0.5 * audio_right
    expected_right = audio_right + 0.5 * audio_left

    assert result.shape == (audio.shape[0], 2)
    assert np.allclose(result[:, 0], expected_left)
    assert np.allclose(result[:, 1], expected_right)
