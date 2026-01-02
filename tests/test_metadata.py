from __future__ import annotations

import base64
from pathlib import Path

import numpy as np
import soundfile as sf
from mutagen.flac import FLAC, Picture
from pytest import LogCaptureFixture

from smartroon.metadata import copy_metadata


def _make_png_bytes() -> bytes:
    """Возвращает минимальную PNG-картинку 1x1."""

    png_base64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO7B"
        "qWcAAAAASUVORK5CYII="
    )
    return base64.b64decode(png_base64)


def test_copy_metadata_flac_basic(tmp_path: Path) -> None:
    sample_rate = 48_000
    src_path = tmp_path / "src.flac"
    dst_path = tmp_path / "dst.flac"

    audio = np.zeros(1_024, dtype=np.float64)
    sf.write(src_path, audio, sample_rate, format="FLAC")
    sf.write(dst_path, audio, sample_rate, format="FLAC")

    flac = FLAC(src_path)
    flac["artist"] = "Test Artist"
    flac["album"] = "Test Album"
    flac["title"] = "Test Title"
    flac["tracknumber"] = "1"
    flac["genre"] = "Test Genre"
    flac["date"] = "2024"

    picture = Picture()
    picture.data = _make_png_bytes()
    picture.mime = "image/png"
    picture.type = 3
    picture.desc = "cover"
    flac.add_picture(picture)
    flac.save()

    copy_metadata(src_path, dst_path)

    copied = FLAC(dst_path)
    assert copied["artist"][0] == "Test Artist"
    assert copied["album"][0] == "Test Album"
    assert copied["title"][0] == "Test Title"
    assert copied["tracknumber"][0] == "1"
    assert copied["genre"][0] == "Test Genre"
    assert copied["date"][0] == "2024"

    assert len(copied.pictures) == 1
    copied_picture = copied.pictures[0]
    assert copied_picture.mime == "image/png"
    assert copied_picture.data == picture.data


def test_copy_metadata_unsupported_format(tmp_path: Path, caplog: LogCaptureFixture) -> None:
    src_path = tmp_path / "src.bin"
    dst_path = tmp_path / "dst.bin"
    src_path.write_bytes(b"not audio")
    dst_path.write_bytes(b"also not audio")

    copy_metadata(src_path, dst_path)

    warnings = [record for record in caplog.records if record.levelname == "WARNING"]
    assert any("Не удалось перенести метаданные" in record.message for record in warnings)
