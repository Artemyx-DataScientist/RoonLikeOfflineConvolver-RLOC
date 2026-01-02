from __future__ import annotations

from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import soundfile as sf

from smartroon.dsp.truepeak import recommended_gain_db, true_peak_db
from smartroon.headroom import analyze_headroom
from smartroon.types import FilterConfig, FilterPath


def _write_wav(path: Path, data: np.ndarray, sample_rate: int) -> None:
    sf.write(path, data, sample_rate, format="WAV", subtype="DOUBLE")


def _build_identity_zip(archive_path: Path, ir_path: str, sample_rate: int) -> None:
    ir_data = np.array([1.0], dtype=np.float64)
    with ZipFile(archive_path, "w") as archive:
        buffer = BytesIO()
        sf.write(buffer, ir_data, sample_rate, format="WAV", subtype="DOUBLE")
        archive.writestr(ir_path, buffer.getvalue())
        cfg = FilterConfig(
            sample_rate=sample_rate,
            num_in=1,
            num_out=1,
            paths=[FilterPath(in_gains=[1.0], out_gains=[1.0], ir_path=ir_path, ir_channel=0)],
        )
        config_content = "\n".join(
            [
                f"{sample_rate} {cfg.num_in} {cfg.num_out} 0",
                "0",
                "0",
                ir_path,
                "0",
                "1",
                "1",
            ]
        )
        archive.writestr("convolver.cfg", config_content)


def test_true_peak_on_unity_signal() -> None:
    signal = np.ones((1024, 2), dtype=np.float64)
    peak = true_peak_db(signal)
    assert np.isclose(peak, 0.0, atol=1e-6)


def test_recommended_gain_negative() -> None:
    gain = recommended_gain_db(peak_db=2.0, target_db=-0.1)
    assert np.isclose(gain, -2.1, atol=1e-6)


def test_analyze_headroom_identity_chain(tmp_path: Path) -> None:
    sample_rate = 48_000
    audio_path = tmp_path / "input.wav"
    archive_path = tmp_path / "filters.zip"

    _write_wav(audio_path, np.array([1.0, -1.0, 0.5], dtype=np.float64), sample_rate)
    _build_identity_zip(archive_path, "ir/identity.wav", sample_rate)

    report = analyze_headroom(archive_path, audio_path)

    assert report["sample_rate"] == sample_rate
    assert np.isclose(report["true_peak_before_db"], 0.0, atol=1e-2)
    assert np.isclose(report["target_true_peak_db"], -0.1, atol=1e-6)
    assert np.isclose(report["recommended_gain_db"], -0.1, atol=1e-2)
    assert np.isclose(report["recommended_headroom_db"], 0.1, atol=1e-2)
