from __future__ import annotations

from pathlib import Path
from typing import Dict
from zipfile import ZipFile

import pytest

from smartroon.loaders import load_filter_from_zip
from smartroon.types import FilterConfig


def _build_cfg_with_kemar_zip(tmp_path: Path) -> Path:
    archive_path = tmp_path / "hybrid.zip"

    cfg_content = "\n".join(
        [
            "44100 1 1 0",
            "0",
            "0",
            "irs/a.wav",
            "0",
            "1",
            "1",
        ]
    )

    kemar_content = "\n".join(
        [
            "48000 1 1",
            "0 0 0 0 kemar/a.wav",
        ]
    )

    with ZipFile(archive_path, "w") as archive:
        archive.writestr("conv/convolver.cfg", cfg_content.encode("utf-8"))
        archive.writestr("conv/irs/a.wav", b"\x00")
        archive.writestr("kemar/config.txt", kemar_content.encode("utf-8"))
        archive.writestr("kemar/kemar/a.wav", b"\x00")

    return archive_path


def _build_kemar_only_zip(tmp_path: Path) -> Path:
    archive_path = tmp_path / "kemar.zip"
    config_content = "\n".join(
        [
            "44100 2 2",
            "0 0 0 0 irs/left.wav",
            "1 1 0 0 irs/right.wav",
        ]
    )

    files: Dict[str, bytes] = {
        "Atmos_KEMAR_v2/config.txt": config_content.encode("utf-8"),
        "Atmos_KEMAR_v2/irs/left.wav": b"\x00\x01",
        "Atmos_KEMAR_v2/irs/right.wav": b"\x00\x01",
    }

    with ZipFile(archive_path, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)

    return archive_path


def test_load_filter_prefers_cfg(tmp_path: Path) -> None:
    archive_path = _build_cfg_with_kemar_zip(tmp_path)

    configs = load_filter_from_zip(archive_path)

    assert set(configs.keys()) == {44100}
    config = configs[44100]
    assert isinstance(config, FilterConfig)
    assert config.paths[0].ir_path == "conv/irs/a.wav"


def test_load_filter_uses_kemar_when_no_cfg(tmp_path: Path) -> None:
    archive_path = _build_kemar_only_zip(tmp_path)

    configs = load_filter_from_zip(archive_path)

    assert set(configs.keys()) == {44100}
    config = configs[44100]
    assert isinstance(config, FilterConfig)
    assert config.paths[0].ir_path == "Atmos_KEMAR_v2/irs/left.wav"


def test_load_filter_raises_for_missing_configs(tmp_path: Path) -> None:
    archive_path = tmp_path / "empty.zip"
    with ZipFile(archive_path, "w"):
        pass

    with pytest.raises(ValueError, match="не найден поддерживаемый конфиг"):
        load_filter_from_zip(archive_path)
