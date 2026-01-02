from __future__ import annotations

from pathlib import Path
from typing import Dict, List
from zipfile import ZipFile

import pytest

from smartroon.loaders import find_cfg, load_all_cfg, load_cfg
from smartroon.types import FilterConfig


def _build_cfg_zip(tmp_path: Path) -> Path:
    archive_path = tmp_path / "convolver.zip"
    cfg_content = "\n".join(
        [
            "# sample rate and channels",
            "44100 2 2 FFFF",
            "1.5 0",
            "0 0",
            "irs/left.wav",
            "0",
            "1 0",
            "0 1",
            "irs/right.wav",
            "1",
            "0 1",
            "1 0",
        ]
    )
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("filters/convolver.cfg", cfg_content.encode("utf-8"))
        archive.writestr("filters/irs/left.wav", b"\x00\x01")
        archive.writestr("filters/irs/right.wav", b"\x00\x01")
    return archive_path


def _build_multi_cfg_zip(tmp_path: Path) -> Path:
    archive_path = tmp_path / "multi.zip"
    cfg_44 = "\n".join(
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
    cfg_48 = "\n".join(
        [
            "48000 1 1 0",
            "0",
            "0",
            "irs48/a.wav",
            "0",
            "1",
            "1",
        ]
    )
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("conv44.cfg", cfg_44.encode("utf-8"))
        archive.writestr("cfgs/conv48.cfg", cfg_48.encode("utf-8"))
        archive.writestr("irs/a.wav", b"\x00")
        archive.writestr("cfgs/irs48/a.wav", b"\x00")
    return archive_path


def _build_missing_ir_zip(tmp_path: Path) -> Path:
    archive_path = tmp_path / "missing.zip"
    cfg_content = "\n".join(
        [
            "44100 1 1 0",
            "0",
            "0",
            "missing.wav",
            "0",
            "1",
            "1",
        ]
    )
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("convolver.cfg", cfg_content.encode("utf-8"))
    return archive_path


def test_find_cfg_lists_all_cfg(tmp_path: Path) -> None:
    archive_path = _build_multi_cfg_zip(tmp_path)
    cfgs = find_cfg(archive_path)
    assert cfgs == ["cfgs/conv48.cfg", "conv44.cfg"]


def test_load_cfg_parses_single_file(tmp_path: Path) -> None:
    archive_path = _build_cfg_zip(tmp_path)

    config = load_cfg(archive_path, "filters/convolver.cfg")
    assert isinstance(config, FilterConfig)
    assert config.sample_rate == 44100
    assert config.num_in == 2
    assert config.num_out == 2
    assert len(config.paths) == 2

    first = config.paths[0]
    assert first.ir_path == "filters/irs/left.wav"
    assert first.ir_channel == 0
    assert first.in_gains == [1.0, 0.0]
    assert first.out_gains == [0.0, 1.0]
    assert first.delay_ms == 1.5


def test_load_all_cfg_merges_by_sample_rate(tmp_path: Path) -> None:
    archive_path = _build_multi_cfg_zip(tmp_path)

    configs = load_all_cfg(archive_path)
    assert set(configs.keys()) == {44100, 48000}
    assert all(isinstance(cfg, FilterConfig) for cfg in configs.values())

    cfg_44 = configs[44100]
    assert cfg_44.paths[0].ir_path == "irs/a.wav"


def test_load_cfg_missing_ir_raises(tmp_path: Path) -> None:
    archive_path = _build_missing_ir_zip(tmp_path)
    with pytest.raises(FileNotFoundError):
        load_cfg(archive_path, "convolver.cfg")
