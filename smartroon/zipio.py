from __future__ import annotations

from pathlib import Path
from typing import List
from zipfile import BadZipFile, ZipFile


def _open_zip(zip_path: Path) -> ZipFile:
    try:
        return ZipFile(zip_path, "r")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"ZIP-архив не найден: {zip_path}") from exc
    except BadZipFile as exc:
        raise BadZipFile(f"Некорректный ZIP-архив: {zip_path}") from exc
    except OSError as exc:
        raise OSError(f"Ошибка чтения ZIP-архива {zip_path}: {exc}") from exc


def list_files(zip_path: Path | str) -> List[str]:
    """Возвращает список файлов внутри ZIP."""

    path = Path(zip_path)
    with _open_zip(path) as archive:
        return archive.namelist()


def read_bytes(zip_path: Path | str, inner_path: Path | str) -> bytes:
    """Читает бинарное содержимое файла внутри ZIP."""

    path = Path(zip_path)
    target = Path(inner_path)
    try:
        with _open_zip(path) as archive:
            with archive.open(str(target)) as handle:
                return handle.read()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"ZIP-архив не найден: {path}") from exc
    except KeyError as exc:
        raise FileNotFoundError(
            f"Файл {target} не найден внутри ZIP-архива {path}"
        ) from exc
    except BadZipFile as exc:
        raise BadZipFile(f"Некорректный ZIP-архив: {path}") from exc
    except OSError as exc:
        raise OSError(f"Ошибка чтения {target} из {path}: {exc}") from exc


def read_text(zip_path: Path | str, inner_path: Path | str, encoding: str = "utf-8") -> str:
    """Читает текстовое содержимое файла внутри ZIP."""

    try:
        data = read_bytes(zip_path, inner_path)
        return data.decode(encoding)
    except UnicodeDecodeError as exc:
        raise UnicodeDecodeError(
            exc.encoding,
            exc.object,
            exc.start,
            exc.end,
            f"Не удалось декодировать {inner_path} из {zip_path}: {exc.reason}",
        ) from exc


def preview_zip(zip_path: Path | str) -> None:
    """Печатает первые файлы и содержимое config.txt для ручной проверки."""

    path = Path(zip_path)
    files = list_files(path)
    print("Первые 10 файлов в архиве:")
    for name in files[:10]:
        print(f"  {name}")

    config_path = Path("Atmos_KEMAR_v2/config.txt")
    if str(config_path) in files:
        print("\nПервые 3 строки Atmos_KEMAR_v2/config.txt:")
        content = read_text(path, config_path)
        for line in content.splitlines()[:3]:
            print(f"  {line}")
    else:
        print("\nФайл Atmos_KEMAR_v2/config.txt не найден в архиве.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Самопроверка чтения ZIP-архива.")
    parser.add_argument("zip_path", type=Path, help="Путь к ZIP-архиву для проверки")
    args = parser.parse_args()
    preview_zip(args.zip_path)
