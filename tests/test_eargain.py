from __future__ import annotations

from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import soundfile as sf

from smartroon.dsp.eargain import apply_ear_gain
from smartroon.headroom import analyze_headroom


def _write_ir(archive: ZipFile, path: str, sample_rate: int) -> None:
    buffer = BytesIO()
    sf.write(buffer, np.array([1.0], dtype=np.float64), sample_rate, format="WAV", subtype="DOUBLE")
    archive.writestr(path, buffer.getvalue())


def _build_stereo_identity_zip(archive_path: Path, sample_rate: int) -> None:
    left_ir_path = "ir/left.wav"
    right_ir_path = "ir/right.wav"
    with ZipFile(archive_path, "w") as archive:
        _write_ir(archive, left_ir_path, sample_rate)
        _write_ir(archive, right_ir_path, sample_rate)
        config_lines = [
            f"{sample_rate} 2 2 0",
            "0 0",
            "0 0",
            left_ir_path,
            "0",
            "1 0",
            "1 0",
            right_ir_path,
            "0",
            "0 1",
            "0 1",
        ]
        archive.writestr("convolver.cfg", "\n".join(config_lines))


def test_ear_gain_identity() -> None:
    audio = np.array([[0.5, -0.5], [1.0, -1.0]], dtype=np.float64)
    output = apply_ear_gain(audio, (0.0, 0.0))
    assert np.allclose(output, audio)


def test_ear_gain_math() -> None:
    t = np.linspace(0, 1, 1024, endpoint=False, dtype=np.float64)
    base = np.sin(2 * np.pi * 440 * t)
    audio = np.stack([base, 0.5 * base], axis=1)

    output = apply_ear_gain(audio, (6.0, 0.0))
    expected_left = audio[:, 0] * (10 ** (6.0 / 20.0))

    assert np.allclose(output[:, 0], expected_left)
    assert np.allclose(output[:, 1], audio[:, 1])


def test_headroom_with_ear_gain(tmp_path: Path) -> None:
    sample_rate = 48_000
    audio_path = tmp_path / "input.wav"
    archive_path = tmp_path / "filters.zip"

    audio = np.full((1_024, 2), 0.8, dtype=np.float64)
    sf.write(audio_path, audio, sample_rate, format="WAV", subtype="DOUBLE")
    _build_stereo_identity_zip(archive_path, sample_rate)

    base_report = analyze_headroom(archive_path, audio_path, target_db=-0.1)
    ear_report = analyze_headroom(
        archive_path, audio_path, target_db=-0.1, ear_gain_left_db=3.0, ear_gain_right_db=0.0
    )

    assert ear_report["recommended_gain_db"] < base_report["recommended_gain_db"]
