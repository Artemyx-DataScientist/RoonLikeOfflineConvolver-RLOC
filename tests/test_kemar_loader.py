from __future__ import annotations

from math import isclose
from pathlib import Path
from typing import Dict
from zipfile import ZipFile

import pytest

from smartroon.loaders import find_kemar_config, load_kemar
from smartroon.types import FilterConfig


def _build_kemar_zip(tmp_path: Path) -> Path:
    archive_path = tmp_path / "kemar.zip"
    config_content = "\n".join(
        [
            "44100 2 2",
            "0 0 0 0 irs/left.wav",
            "1 0 1.5 -3 irs/left.wav",
            "0 1 0.5 0 irs/right.wav",
            "1 1 0 -1 irs/right.wav",
            "48000 2 2",
            "0 0 0 0 irs48/left.wav",
            "1 1 0 0 irs48/right.wav",
        ]
    )
    files: Dict[str, bytes] = {
        "Atmos_KEMAR_v2/config.txt": config_content.encode("utf-8"),
        "Atmos_KEMAR_v2/irs/left.wav": b"\x00\x01",
        "Atmos_KEMAR_v2/irs/right.wav": b"\x00\x01",
        "Atmos_KEMAR_v2/irs48/left.wav": b"\x00\x01",
        "Atmos_KEMAR_v2/irs48/right.wav": b"\x00\x01",
    }
    with ZipFile(archive_path, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return archive_path


def _build_broken_zip(tmp_path: Path) -> Path:
    archive_path = tmp_path / "broken.zip"
    config_content = "\n".join(
        [
            "44100 1 1",
            "0 0 0 0 missing.wav",
        ]
    )
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("folder/config.txt", config_content.encode("utf-8"))
    return archive_path


def test_find_kemar_config_locates_config(tmp_path: Path) -> None:
    zip_path = _build_kemar_zip(tmp_path)
    config = find_kemar_config(zip_path)
    assert config == "Atmos_KEMAR_v2/config.txt"


def test_load_kemar_parses_blocks(tmp_path: Path) -> None:
    zip_path = _build_kemar_zip(tmp_path)

    configs = load_kemar(zip_path)
    assert set(configs.keys()) == {44100, 48000}

    config_44 = configs[44100]
    assert isinstance(config_44, FilterConfig)
    assert config_44.num_in == 2
    assert config_44.num_out == 2
    assert len(config_44.paths) == 4

    first = config_44.paths[0]
    assert first.in_gains == [1.0, 0.0]
    assert first.out_gains == [1.0, 0.0]
    assert first.ir_path == "Atmos_KEMAR_v2/irs/left.wav"
    assert isclose(first.delay_ms, 0.0)

    second = config_44.paths[1]
    assert second.in_gains == [0.0, 1.0]
    assert isclose(second.out_gains[0], 10 ** (-3 / 20))
    assert second.out_gains[1] == 0.0
    assert second.ir_path == "Atmos_KEMAR_v2/irs/left.wav"
    assert isclose(second.delay_ms, 1.5)


def test_load_kemar_raises_for_missing_ir(tmp_path: Path) -> None:
    zip_path = _build_broken_zip(tmp_path)
    with pytest.raises(FileNotFoundError):
        load_kemar(zip_path)
