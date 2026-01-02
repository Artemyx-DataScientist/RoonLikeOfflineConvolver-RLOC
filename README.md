# RoonLikeOfflineConvolver-RLOC

Утилиты для офлайн-конволюции и автоматического нормирования громкости при экспорте аудиотеки на телефон. Поддерживаются архивы с импульсными характеристиками в форматах Convolver (`.cfg`) и KEMAR (`config.txt`), а также готовые пресеты для Roon.

## Требования

- Python 3.10+.
- Зависимости для аудиообработки: `numpy`, `scipy`, `soundfile`.
- Для запуска тестов понадобится `pytest`.

Рекомендуется работать в виртуальном окружении:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -U pip
pip install numpy scipy soundfile pytest
```

## Структура архивов с фильтрами

- **Convolver (`*.cfg`)**: в архиве присутствуют один или несколько файлов `*.cfg`. Каждый содержит блоки с частотой дискретизации, числами входных/выходных каналов, задержками и наборами путей к IR. Путь к IR можно указывать относительно расположения `.cfg` внутри ZIP.
- **KEMAR (`config.txt`)**: в архиве есть `config.txt`. Для каждого sample rate задаются пары вход→выход с задержкой, усилением в дБ и относительным путём к IR.

Библиотека автоматически выбирает подходящий формат: сначала ищет `.cfg`, затем `config.txt`. Для загрузки конфигураций можно использовать `smartroon.load_filter_from_zip`, который вернёт словарь `sample_rate -> FilterConfig`.

## CLI

Запуск через модуль:

```bash
python -m smartroon --help
```

Общий аргумент:
- `--inspect-zip ZIP` — вывести список первых файлов из архива и первые строки `Atmos_KEMAR_v2/config.txt`.

Доступные команды:

### `headroom`

Рассчитывает true peak после конволюции и рекомендуемый headroom.

```bash
python -m smartroon headroom \
  --audio /path/to/input.wav \
  --filter-zip /path/to/filters.zip \
  --target-tp -0.1 \
  --oversample 4 \
  --json report.json
```

Ключевые параметры:
- `--audio PATH` — входной WAV/FLAC/…; число каналов должно соответствовать конфигурации фильтра.
- `--filter-zip ZIP` — архив с IR.
- `--target-tp DBFS` — целевой true peak (по умолчанию `-0.1` dBFS).
- `--oversample N` — фактор оверсемплинга для расчёта true peak (по умолчанию `4`).
- `--json OUT.json` — сохранить отчёт.

### `render`

Выполняет конволюцию, применяет рекомендуемый gain и записывает WAV (`PCM_24`).

```bash
python -m smartroon render \
  --audio /path/to/input.wav \
  --filter-zip /path/to/filters.zip \
  --output /path/to/output.wav \
  --target-tp -0.1 \
  --oversample 4 \
  --json render_report.json
```

### `verify`

Создаёт контрольные артефакты для выбранного фрагмента: входной и выходной сниппеты, а также `report.json` с метриками RMS, true peak и SHA256.

```bash
python -m smartroon verify \
  --audio /path/to/input.wav \
  --filter-zip /path/to/filters.zip \
  --seconds 5 \
  --output-dir verify_artifacts
```

Параметры:
- `--seconds N` — длительность анализируемого фрагмента (по умолчанию `5` секунд).
- `--output-dir DIR` — каталог для `snippet_in.wav`, `snippet_out.wav` и `report.json`. Если не указан, создаётся `<input>_verify`.

## Программный интерфейс

- `smartroon.load_filter_from_zip(zip_path)` — определить формат архива и получить `FilterConfig` по частоте дискретизации.
- `smartroon.dsp.convolver.convolve(audio, sr, cfg, zip_path)` — выполнить конволюцию в numpy.
- `smartroon.headroom.analyze_headroom(...)` и `smartroon.headroom.render_convolved(...)` — вычислить рекомендуемый gain и сохранить обработанный файл.
- `smartroon.verify.run_verify(...)` — сформировать контрольные артефакты и отчёт.

Все функции принимают `pathlib.Path` или строки; для путей предпочтительнее использовать `Path`.

## Тесты

Для прогонки всех проверок из репозитория:

```bash
python -m pytest
```
