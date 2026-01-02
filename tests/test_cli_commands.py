from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Tuple
from zipfile import ZipFile

import numpy as np
import soundfile as sf
from pytest import CaptureFixture

from smartroon.cli import main
from smartroon.dsp.truepeak import true_peak_db
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


def _prepare_inputs(tmp_path: Path) -> Tuple[Path, Path, int]:
    sample_rate = 48_000
    audio_path = tmp_path / "input.wav"
    archive_path = tmp_path / "filters.zip"
    samples = np.linspace(-1.0, 1.0, sample_rate * 2, dtype=np.float64)
    _write_wav(audio_path, samples, sample_rate)
    _build_identity_zip(archive_path, "ir/identity.wav", sample_rate)
    return audio_path, archive_path, sample_rate


def test_headroom_cli_outputs_report(capsys: CaptureFixture[str], tmp_path: Path) -> None:
    audio_path, archive_path, _ = _prepare_inputs(tmp_path)
    json_path = tmp_path / "report.json"

    main(
        [
            "headroom",
            "--audio",
            str(audio_path),
            "--filter-zip",
            str(archive_path),
            "--json",
            str(json_path),
        ]
    )

    captured = capsys.readouterr().out
    assert "Recommended gain" in captured
    assert json_path.exists()


def test_render_cli_writes_output(tmp_path: Path) -> None:
    audio_path, archive_path, sample_rate = _prepare_inputs(tmp_path)
    output_path = tmp_path / "out.wav"

    main(
        [
            "render",
            "--audio",
            str(audio_path),
            "--filter-zip",
            str(archive_path),
            "--output",
            str(output_path),
            "--target-tp",
            "-0.5",
        ]
    )

    assert output_path.exists()
    rendered, sr = sf.read(output_path, always_2d=False, dtype="float64")
    assert sr == sample_rate
    peak_after = true_peak_db(rendered, oversample=4)
    assert peak_after <= -0.5 + 1e-6


def test_render_stream_only_with_manual_gain(tmp_path: Path) -> None:
    audio_path, archive_path, sample_rate = _prepare_inputs(tmp_path)
    output_path = tmp_path / "out_stream.wav"

    main(
        [
            "render",
            "--audio",
            str(audio_path),
            "--filter-zip",
            str(archive_path),
            "--output",
            str(output_path),
            "--stream-only",
            "--gain-db",
            "-6.0",
            "--chunk-size",
            "2048",
        ]
    )

    assert output_path.exists()
    rendered, sr = sf.read(output_path, always_2d=False, dtype="float64")
    assert sr == sample_rate
    # Исходные данные линейно возрастают от -1 до 1, IR=identity, поэтому gain должен сработать.
    assert np.isclose(float(rendered.max()), 1.0 * (10 ** (-6.0 / 20.0)), atol=1e-6)


def test_verify_cli_saves_artifacts(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    audio_path, archive_path, sample_rate = _prepare_inputs(tmp_path)
    output_dir = tmp_path / "verify_artifacts"

    main(
        [
            "verify",
            "--audio",
            str(audio_path),
            "--filter-zip",
            str(archive_path),
            "--seconds",
            "1",
            "--output-dir",
            str(output_dir),
        ]
    )

    snippet_in = output_dir / "snippet_in.wav"
    snippet_out = output_dir / "snippet_out.wav"
    report_path = output_dir / "report.json"

    assert snippet_in.exists()
    assert snippet_out.exists()
    assert report_path.exists()

    snippet_data, sr_in = sf.read(snippet_in, always_2d=True, dtype="float64")
    snippet_out_data, sr_out = sf.read(snippet_out, always_2d=True, dtype="float64")
    assert sr_in == sample_rate
    assert sr_out == sample_rate
    assert snippet_data.shape[0] == 48_000

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["sample_rate"] == sample_rate
    assert report["input_frames"] == 48_000
    assert report["ir_lengths"] == [1]
    assert snippet_out_data.shape[0] == snippet_data.shape[0] + report["ir_lengths"][0] - 1
    assert report["output"]["checksum"] == report["input"]["checksum"]

    captured = capsys.readouterr().out
    assert "RMS per channel" in captured
    assert "True peak" in captured
