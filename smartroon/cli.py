from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

from . import __version__
from .headroom import analyze_headroom, render_convolved
from .zipio import preview_zip


def _positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("значение должно быть положительным целым числом")
    return result


def _add_common_io_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--audio",
        required=True,
        type=Path,
        metavar="PATH",
        help="Путь к входному аудиофайлу",
    )
    parser.add_argument(
        "--filter-zip",
        required=True,
        type=Path,
        metavar="ZIP",
        help="Путь к ZIP с фильтрами",
    )
    parser.add_argument(
        "--target-tp",
        type=float,
        default=-0.1,
        metavar="DBFS",
        help="Целевой true peak (по умолчанию -0.1 dBFS)",
    )
    parser.add_argument(
        "--oversample",
        type=_positive_int,
        default=4,
        metavar="N",
        help="Фактор оверсемплинга (по умолчанию 4)",
    )


def _build_headroom_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    headroom_parser = subparsers.add_parser(
        "headroom",
        help="Оценить true peak и рекомендуемый headroom",
        description="Конволюция входного файла фильтром и расчёт рекомендуемого headroom.",
    )
    _add_common_io_arguments(headroom_parser)
    headroom_parser.add_argument(
        "--json",
        type=Path,
        metavar="OUT.json",
        help="Сохранить отчёт в JSON",
    )


def _build_render_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    render_parser = subparsers.add_parser(
        "render",
        help="Выполнить конволюцию, применить gain и записать WAV",
        description="Конволюция входного файла, расчёт gain и сохранение результата.",
    )
    _add_common_io_arguments(render_parser)
    render_parser.add_argument(
        "--json",
        type=Path,
        metavar="OUT.json",
        help="Сохранить отчёт в JSON",
    )
    render_parser.add_argument(
        "--output",
        required=True,
        type=Path,
        metavar="OUT.wav",
        help="Путь для сохранения результирующего WAV (PCM_24)",
    )


def build_parser() -> argparse.ArgumentParser:
    """Создает корневой парсер CLI."""

    parser = argparse.ArgumentParser(
        prog="smartroon",
        description="Инструменты SmartRoon (консольные команды).",
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

    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    _build_headroom_parser(subparsers)
    _build_render_parser(subparsers)

    return parser


def run_inspect(zip_path: Path) -> None:
    """Запускает просмотр содержимого ZIP."""

    preview_zip(zip_path)


def _format_report(report: Dict[str, float | int | str]) -> str:
    lines: List[str] = [
        f"Sample rate: {report.get('sample_rate')}",
        f"Target true peak: {report.get('target_true_peak_db')} dBFS",
        f"True peak before: {report.get('true_peak_before_db')} dBFS",
        f"Recommended gain: {report.get('recommended_gain_db')} dB",
        f"Recommended headroom: {report.get('recommended_headroom_db')} dB",
    ]
    if "true_peak_after_db" in report:
        lines.append(f"True peak after: {report.get('true_peak_after_db')} dBFS")
    if "output_path" in report:
        lines.append(f"Output file: {report.get('output_path')}")
    return "\n".join(lines)


def _write_json(path: Path, report: Dict[str, float | int | str]) -> None:
    import json

    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)


def run_headroom(args: argparse.Namespace) -> int:
    """Запускает команду headroom."""

    report = analyze_headroom(
        zip_path=args.filter_zip,
        audio_path=args.audio,
        target_db=args.target_tp,
        oversample=args.oversample,
    )
    print(_format_report(report))
    if args.json:
        _write_json(args.json, report)
    return 0


def run_render(args: argparse.Namespace) -> int:
    """Запускает команду render."""

    report = render_convolved(
        zip_path=args.filter_zip,
        audio_path=args.audio,
        output_path=args.output,
        target_db=args.target_tp,
        oversample=args.oversample,
    )
    print(_format_report(report))
    if args.json:
        _write_json(args.json, report)
    return 0


def _dispatch(args: argparse.Namespace) -> int:
    if args.inspect_zip:
        run_inspect(args.inspect_zip)
        return 0
    if args.command == "headroom":
        return run_headroom(args)
    if args.command == "render":
        return run_render(args)
    return 1


def main(argv: Optional[List[str]] = None) -> None:
    """Точка входа CLI."""

    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.inspect_zip and not args.command:
        parser.print_help()
        return

    try:
        exit_code = _dispatch(args)
    except Exception as exc:  # noqa: BLE001
        parser.error(str(exc))
        return

    if exit_code != 0:
        parser.exit(exit_code)


if __name__ == "__main__":
    main()
