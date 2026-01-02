from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

from . import __version__
from .zipio import preview_zip


def build_parser() -> argparse.ArgumentParser:
    """Создает корневой парсер CLI."""

    parser = argparse.ArgumentParser(
        prog="smartroon",
        description="Инструменты SmartRoon (предварительный каркас CLI).",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
        help="Показать версию",
    )
    parser.add_argument(
        "--inspect-zip",
        type=Path,
        metavar="ZIP",
        help="Вывести первые файлы из архива и первые строки Atmos_KEMAR_v2/config.txt",
    )
    return parser


def run_inspect(zip_path: Path) -> None:
    """Запускает просмотр содержимого ZIP."""

    preview_zip(zip_path)


def main(argv: Optional[List[str]] = None) -> None:
    """Точка входа CLI."""

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.inspect_zip:
        run_inspect(args.inspect_zip)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
