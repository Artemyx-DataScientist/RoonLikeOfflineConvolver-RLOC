from __future__ import annotations

from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import soundfile as sf

from smartroon.dsp.convolver import convolve
from smartroon.dsp.streaming_convolver import stream_convolve_to_file
from smartroon.types import FilterConfig, FilterPath


def _write_ir_wav(ir_data: np.ndarray, sample_rate: int) -> bytes:
    buffer = BytesIO()
    sf.write(buffer, ir_data, sample_rate, format="WAV", subtype="DOUBLE")
    return buffer.getvalue()


def test_streaming_matches_full_convolution(tmp_path: Path) -> None:
    sample_rate = 48_000
    ir_path = "irs/test_ir.wav"

    rng = np.random.default_rng(123)
    audio = (rng.standard_normal(10_000) * 0.01).astype(np.float64)
    ir = rng.standard_normal(1_024).astype(np.float64)

    audio_path = tmp_path / "audio.wav"
    sf.write(audio_path, audio, sample_rate, format="WAV", subtype="DOUBLE")

    archive_path = tmp_path / "filters.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr(ir_path, _write_ir_wav(ir, sample_rate))

    config = FilterConfig(
        sample_rate=sample_rate,
        num_in=1,
        num_out=1,
        paths=[FilterPath(in_gains=[1.0], out_gains=[1.0], ir_path=ir_path, ir_channel=0)],
    )

    expected = convolve(audio, sample_rate, config, archive_path)[:, 0]

    output_path = tmp_path / "stream_out.wav"
    stream_convolve_to_file(
        audio_path=audio_path,
        zip_path=archive_path,
        cfg=config,
        output_path=output_path,
        chunk_size=2_048,
        progress=False,
        output_subtype="DOUBLE",
    )

    rendered, sr_out = sf.read(output_path, always_2d=False, dtype="float64")
    assert sr_out == sample_rate
    assert rendered.shape == expected.shape
    assert np.allclose(rendered, expected, rtol=1e-6, atol=1e-6)
