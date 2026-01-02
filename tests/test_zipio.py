from __future__ import annotations

from pathlib import Path
from typing import List
from zipfile import ZipFile

import pytest

from smartroon.zipio import list_files, preview_zip, read_bytes, read_text


def _create_sample_zip(tmp_path: Path) -> Path:
    archive_path = tmp_path / "sample.zip"
    files: List[tuple[str, bytes]] = [
        ("file1.txt", b"first"),
        ("file2.txt", b"second"),
        ("dir/file3.bin", b"\x00\x01"),
        ("Atmos_KEMAR_v2/config.txt", b"line1\nline2\nline3\nline4"),
    ]
    with ZipFile(archive_path, "w") as archive:
        for name, content in files:
            archive.writestr(name, content)
    return archive_path


def test_list_files_reads_archive(tmp_path: Path) -> None:
    zip_path = _create_sample_zip(tmp_path)
    names = list_files(zip_path)
    assert "file1.txt" in names
    assert "dir/file3.bin" in names
    assert "Atmos_KEMAR_v2/config.txt" in names


def test_read_text_and_bytes(tmp_path: Path) -> None:
    zip_path = _create_sample_zip(tmp_path)
    assert read_bytes(zip_path, "file1.txt") == b"first"
    assert read_text(zip_path, "Atmos_KEMAR_v2/config.txt").startswith("line1\nline2\nline3")


def test_preview_zip_outputs_expected_lines(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    zip_path = _create_sample_zip(tmp_path)
    preview_zip(zip_path)
    captured = capsys.readouterr().out
    assert "Первые 10 файлов" in captured
    assert "file1.txt" in captured
    assert "Atmos_KEMAR_v2/config.txt" in captured
    assert "line1" in captured and "line2" in captured and "line3" in captured
