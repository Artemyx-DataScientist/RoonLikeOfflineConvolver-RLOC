"""Main window for the RLOC GUI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QProgressBar,
    QRadioButton,
    QMessageBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtWidgets import QHeaderView

from smartroon import FilterConfig, load_filter_from_zip
from smartroon.headroom import analyze_headroom


@dataclass(frozen=True)
class _HeadroomTask:
    row_index: int
    path: Path


@dataclass(frozen=True)
class _HeadroomResult:
    row_index: int
    true_peak_db: float | None
    recommended_gain_db: float | None
    status: str
    error_message: str | None = None


class _HeadroomWorker(QObject):
    result_ready = Signal(object)
    finished = Signal()
    log_message = Signal(str)

    def __init__(
        self,
        filter_path: Path,
        tasks: Sequence[_HeadroomTask],
        target_tp: float,
        oversample: int,
        selected_sample_rate: int,
    ) -> None:
        super().__init__()
        self._filter_path = filter_path
        self._tasks = tasks
        self._target_tp = target_tp
        self._oversample = oversample
        self._selected_sample_rate = selected_sample_rate

    def run(self) -> None:
        for task in self._tasks:
            try:
                self.log_message.emit(f"Анализируем: {task.path}")
                report = analyze_headroom(
                    zip_path=self._filter_path,
                    audio_path=task.path,
                    target_db=self._target_tp,
                    oversample=self._oversample,
                )
                sample_rate = int(report.get("sample_rate", 0))
                if sample_rate != self._selected_sample_rate:
                    raise ValueError(
                        f"Sample rate трека {sample_rate} не совпадает с выбранным SR-конфигом "
                        f"{self._selected_sample_rate}"
                    )

                result = _HeadroomResult(
                    row_index=task.row_index,
                    true_peak_db=float(report["true_peak_before_db"]),
                    recommended_gain_db=float(report["recommended_gain_db"]),
                    status="analyzed",
                )
            except Exception as exc:  # noqa: BLE001
                result = _HeadroomResult(
                    row_index=task.row_index,
                    true_peak_db=None,
                    recommended_gain_db=None,
                    status="error",
                    error_message=str(exc),
                )
            self.result_ready.emit(result)

        self.finished.emit()


class MainWindow(QMainWindow):
    """Main application window with controls and placeholders."""

    _AUDIO_EXTENSIONS: tuple[str, ...] = (".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav")

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("RLOC GUI")
        self.filter_input: QLineEdit | None = None
        self.sample_rate_dropdown: QComboBox | None = None
        self.filter_info_label: QLabel | None = None
        self._log_view: QTextEdit | None = None
        self.progress_bar: QProgressBar | None = None
        self._tracks_table: QTableWidget | None = None
        self._known_files: set[str] = set()
        self._target_tp_spin: QDoubleSpinBox | None = None
        self._oversample_combo: QComboBox | None = None
        self._analysis_thread: QThread | None = None
        self._analysis_worker: _HeadroomWorker | None = None
        self._analysis_total: int = 0
        self._analysis_completed: int = 0
        self._analyze_button: QPushButton | None = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        central_widget = QWidget(self)
        main_layout = QVBoxLayout(central_widget)

        content_layout = QHBoxLayout()
        settings_panel = self._create_settings_panel()
        self._tracks_table = self._create_tracks_table()
        content_layout.addWidget(settings_panel, 0)
        content_layout.addWidget(self._tracks_table, 1)
        content_layout.setStretch(0, 0)
        content_layout.setStretch(1, 1)
        main_layout.addLayout(content_layout)

        self.progress_bar = self._create_progress_bar()
        main_layout.addWidget(self.progress_bar)

        self._log_view = self._create_log_view()
        main_layout.addWidget(self._log_view)

        self.setCentralWidget(central_widget)

    def _create_settings_panel(self) -> QGroupBox:
        settings_group = QGroupBox("Настройки")
        settings_layout = QVBoxLayout(settings_group)

        source_group = self._create_source_group()
        filter_group = self._create_filter_group()
        parameters_group = self._create_parameters_group()
        ear_gain_group = self._create_ear_gain_group()
        output_group = self._create_output_group()
        analyze_button = QPushButton("Проанализировать")
        analyze_button.clicked.connect(self._start_headroom_analysis)
        self._analyze_button = analyze_button

        settings_layout.addWidget(source_group)
        settings_layout.addWidget(filter_group)
        settings_layout.addWidget(parameters_group)
        settings_layout.addWidget(ear_gain_group)
        settings_layout.addWidget(output_group)
        settings_layout.addWidget(analyze_button)
        settings_layout.addStretch(1)

        return settings_group

    def _create_source_group(self) -> QGroupBox:
        group = QGroupBox("Источник")
        layout = QHBoxLayout(group)

        select_files_button = QPushButton("Выбрать файлы")
        select_files_button.clicked.connect(self._select_files)

        select_folder_button = QPushButton("Выбрать папку")
        select_folder_button.clicked.connect(self._select_folder)

        layout.addWidget(select_files_button)
        layout.addWidget(select_folder_button)

        return group

    def _create_filter_group(self) -> QGroupBox:
        group = QGroupBox("Фильтр")
        layout = QHBoxLayout(group)

        self.filter_input = QLineEdit()
        self.filter_input.setPlaceholderText("filter.zip")

        self.sample_rate_dropdown = QComboBox()
        self.sample_rate_dropdown.setPlaceholderText("sample rate")
        self.sample_rate_dropdown.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)

        self.filter_info_label = QLabel("Нет загруженного фильтра")
        self.filter_info_label.setObjectName("filterInfoLabel")

        browse_button = QPushButton("Обзор")
        browse_button.clicked.connect(self._on_browse_filter)

        layout.addWidget(self.filter_input)
        layout.addWidget(self.sample_rate_dropdown)
        layout.addWidget(self.filter_info_label)
        layout.addWidget(browse_button)

        return group

    def _on_browse_filter(self) -> None:
        initial_dir = self._current_filter_dir()
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выбрать filter.zip",
            initial_dir,
            "Zip archives (*.zip);;All files (*)",
        )
        if not file_path:
            return

        if self.filter_input is not None:
            self.filter_input.setText(file_path)

        self._load_filter_configs(Path(file_path))

    def _current_filter_dir(self) -> str:
        if self.filter_input is not None:
            current_text = self.filter_input.text().strip()
            if current_text:
                current_path = Path(current_text).expanduser()
                if current_path.exists():
                    if current_path.is_file():
                        return str(current_path.parent)
                    return str(current_path)
        return str(Path.home())

    def _load_filter_configs(self, zip_path: Path) -> None:
        if self.sample_rate_dropdown is None or self.filter_info_label is None:
            return

        try:
            self._log_message(f"Загружаем фильтр из {zip_path}")
            configs: dict[int, FilterConfig] = load_filter_from_zip(zip_path)
        except Exception as exc:
            self._log_message(f"Ошибка загрузки фильтра: {exc}")
            self.sample_rate_dropdown.clear()
            self.filter_info_label.setText("Ошибка загрузки")
            QMessageBox.critical(
                self,
                "Ошибка фильтра",
                f"Не удалось загрузить фильтр:\n{exc}",
            )
            return

        sample_rates = sorted(configs.keys())
        self.sample_rate_dropdown.clear()
        for rate in sample_rates:
            self.sample_rate_dropdown.addItem(str(rate))

        self.filter_info_label.setText(f"Найдено конфигов: {len(configs)}")
        self._log_message(
            f"Фильтр загружен: sample rates {', '.join(map(str, sample_rates))}"
        )

    def _create_parameters_group(self) -> QGroupBox:
        group = QGroupBox("Параметры")
        layout = QVBoxLayout(group)

        target_tp_layout = QHBoxLayout()
        target_label = QLabel("target_tp")
        target_tp_spin = QDoubleSpinBox()
        target_tp_spin.setRange(-60.0, 6.0)
        target_tp_spin.setSingleStep(0.1)
        target_tp_spin.setValue(-0.1)
        self._target_tp_spin = target_tp_spin
        target_tp_layout.addWidget(target_label)
        target_tp_layout.addWidget(target_tp_spin)

        oversample_layout = QHBoxLayout()
        oversample_label = QLabel("oversample")
        oversample_combo = QComboBox()
        oversample_combo.addItems(["4", "8"])
        self._oversample_combo = oversample_combo
        oversample_layout.addWidget(oversample_label)
        oversample_layout.addWidget(oversample_combo)

        gain_mode_layout = QHBoxLayout()
        gain_mode_label = QLabel("Режим gain")
        down_only_radio = QRadioButton("down-only")
        normalize_radio = QRadioButton("normalize (allow boost)")
        down_only_radio.setChecked(True)
        gain_button_group = QButtonGroup(group)
        gain_button_group.addButton(down_only_radio)
        gain_button_group.addButton(normalize_radio)
        gain_mode_layout.addWidget(gain_mode_label)
        gain_mode_layout.addWidget(down_only_radio)
        gain_mode_layout.addWidget(normalize_radio)

        layout.addLayout(target_tp_layout)
        layout.addLayout(oversample_layout)
        layout.addLayout(gain_mode_layout)

        return group

    def _create_ear_gain_group(self) -> QGroupBox:
        group = QGroupBox("Ear gain")
        layout = QVBoxLayout(group)

        left_layout = QHBoxLayout()
        left_label = QLabel("left dB")
        left_spin = QDoubleSpinBox()
        left_spin.setRange(-24.0, 24.0)
        left_spin.setSingleStep(0.1)
        left_layout.addWidget(left_label)
        left_layout.addWidget(left_spin)

        right_layout = QHBoxLayout()
        right_label = QLabel("right dB")
        right_spin = QDoubleSpinBox()
        right_spin.setRange(-24.0, 24.0)
        right_spin.setSingleStep(0.1)
        right_layout.addWidget(right_label)
        right_layout.addWidget(right_spin)

        offset_layout = QHBoxLayout()
        offset_label = QLabel("ear_offset_db")
        offset_spin = QDoubleSpinBox()
        offset_spin.setRange(-24.0, 24.0)
        offset_spin.setSingleStep(0.1)
        offset_layout.addWidget(offset_label)
        offset_layout.addWidget(offset_spin)

        layout.addLayout(left_layout)
        layout.addLayout(right_layout)
        layout.addLayout(offset_layout)

        return group

    def _create_output_group(self) -> QGroupBox:
        group = QGroupBox("Выход")
        layout = QVBoxLayout(group)

        output_path_layout = QHBoxLayout()
        output_label = QLabel("Папка")
        output_input = QLineEdit()
        browse_button = QPushButton("Обзор")
        browse_button.clicked.connect(lambda: self._log_message("TODO: выбрать папку выгрузки"))
        output_path_layout.addWidget(output_label)
        output_path_layout.addWidget(output_input)
        output_path_layout.addWidget(browse_button)

        keep_structure_checkbox = QCheckBox("сохранять структуру подпапок")

        layout.addLayout(output_path_layout)
        layout.addWidget(keep_structure_checkbox)

        return group

    def _create_tracks_table(self) -> QTableWidget:
        table = QTableWidget()
        columns = [
            "file",
            "sr/ch",
            "duration",
            "true_peak_db",
            "rec_gain_db",
            "status",
        ]
        table.setColumnCount(len(columns))
        table.setHorizontalHeaderLabels(columns)
        table.verticalHeader().setVisible(False)
        table.setRowCount(0)

        header: QHeaderView = table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        header.setStretchLastSection(True)

        placeholder_item = QTableWidgetItem("")
        table.setItemPrototype(placeholder_item)

        return table

    def _create_progress_bar(self) -> QProgressBar:
        progress_bar = QProgressBar()
        progress_bar.setRange(0, 100)
        progress_bar.setValue(0)
        return progress_bar

    def _create_log_view(self) -> QTextEdit:
        log_view = QTextEdit()
        log_view.setReadOnly(True)
        log_view.setPlaceholderText("Лог выполнения")
        return log_view

    def _log_message(self, message: str) -> None:
        if self._log_view is None:
            return
        current_text = self._log_view.toPlainText()
        next_text = f"{current_text}\n{message}" if current_text else message
        self._log_view.setPlainText(next_text)
        self._log_view.moveCursor(QTextCursor.End)

    def _select_files(self) -> None:
        selected_files, _ = QFileDialog.getOpenFileNames(
            self,
            "Выбрать аудиофайлы",
            str(Path.home()),
            self._build_audio_name_filter(),
        )
        if not selected_files:
            return

        self._add_tracks(Path(path) for path in selected_files)

    def _select_folder(self) -> None:
        selected_directory = QFileDialog.getExistingDirectory(self, "Выбрать папку с аудио", str(Path.home()))
        if not selected_directory:
            return

        directory_path = Path(selected_directory)
        audio_files = self._collect_audio_files(directory_path)
        self._add_tracks(audio_files)

    def _collect_audio_files(self, directory: Path) -> list[Path]:
        return [
            path
            for path in directory.rglob("*")
            if path.is_file() and self._is_supported_extension(path)
        ]

    def _add_tracks(self, paths: Iterable[Path]) -> None:
        if self._tracks_table is None:
            return

        new_files: list[Path] = []
        for path in paths:
            resolved_path = path.resolve()
            if not self._is_supported_extension(resolved_path):
                continue

            path_str = str(resolved_path)
            if path_str in self._known_files:
                continue

            self._known_files.add(path_str)
            new_files.append(resolved_path)

        if not new_files:
            return

        table = self._tracks_table
        start_row = table.rowCount()
        table.setUpdatesEnabled(False)
        table.setRowCount(start_row + len(new_files))

        for offset, path in enumerate(new_files):
            row_index = start_row + offset
            table.setItem(row_index, 0, QTableWidgetItem(str(path)))
            for column_index in range(1, 5):
                table.setItem(row_index, column_index, QTableWidgetItem(""))
            table.setItem(row_index, 5, QTableWidgetItem("queued"))

        table.setUpdatesEnabled(True)

    def _is_supported_extension(self, path: Path) -> bool:
        return path.suffix.lower() in self._AUDIO_EXTENSIONS

    def _build_audio_name_filter(self) -> str:
        extensions = " ".join(f"*{ext}" for ext in self._AUDIO_EXTENSIONS)
        return f"Аудиофайлы ({extensions})"

    def _collect_analysis_tasks(self) -> list[_HeadroomTask]:
        if self._tracks_table is None:
            return []

        tasks: list[_HeadroomTask] = []
        for row_index in range(self._tracks_table.rowCount()):
            file_item = self._tracks_table.item(row_index, 0)
            if file_item is None:
                continue
            path_text = file_item.text().strip()
            if not path_text:
                continue
            tasks.append(_HeadroomTask(row_index=row_index, path=Path(path_text)))
        return tasks

    def _start_headroom_analysis(self) -> None:
        if self._analysis_thread is not None and self._analysis_thread.isRunning():
            self._log_message("Анализ уже выполняется")
            return

        if (
            self.filter_input is None
            or self.sample_rate_dropdown is None
            or self._target_tp_spin is None
            or self._oversample_combo is None
        ):
            return

        filter_path_text = self.filter_input.text().strip()
        if not filter_path_text:
            self._log_message("Укажите путь к filter.zip перед анализом")
            QMessageBox.warning(self, "Фильтр не выбран", "Пожалуйста, выберите файл filter.zip.")
            return

        filter_path = Path(filter_path_text).expanduser()
        if not filter_path.exists():
            self._log_message(f"Файл фильтра не найден: {filter_path}")
            QMessageBox.critical(self, "Фильтр не найден", f"Не удалось найти {filter_path}")
            return

        if self.sample_rate_dropdown.currentIndex() < 0:
            self._log_message("Не выбран SR-конфиг")
            QMessageBox.warning(self, "SR-конфиг", "Выберите sample rate конфигурацию.")
            return

        try:
            selected_sample_rate = int(self.sample_rate_dropdown.currentText())
        except ValueError:
            self._log_message("Некорректное значение sample rate в конфиге")
            QMessageBox.critical(self, "Ошибка SR", "Текущее значение sample rate не является числом.")
            return

        oversample_text = self._oversample_combo.currentText()
        try:
            oversample = int(oversample_text)
        except ValueError:
            self._log_message(f"Некорректный oversample: {oversample_text}")
            QMessageBox.critical(self, "Ошибка параметров", "Oversample должен быть числом.")
            return
        if oversample <= 0:
            self._log_message("Oversample должен быть положительным")
            QMessageBox.critical(self, "Ошибка параметров", "Oversample должен быть больше нуля.")
            return

        target_tp = float(self._target_tp_spin.value())
        tasks = self._collect_analysis_tasks()
        if not tasks:
            self._log_message("Нет треков для анализа")
            return

        self._analysis_total = len(tasks)
        self._analysis_completed = 0
        self._update_progress_bar()
        self._set_rows_status(tasks, "analyzing")
        if self._analyze_button is not None:
            self._analyze_button.setEnabled(False)

        worker = _HeadroomWorker(
            filter_path=filter_path,
            tasks=tasks,
            target_tp=target_tp,
            oversample=oversample,
            selected_sample_rate=selected_sample_rate,
        )
        thread = QThread(self)
        worker.moveToThread(thread)

        worker.result_ready.connect(self._handle_analysis_result)
        worker.log_message.connect(self._log_message)
        worker.finished.connect(self._handle_analysis_finished)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.started.connect(worker.run)

        self._analysis_thread = thread
        self._analysis_worker = worker
        thread.start()
        self._log_message("Запущен анализ headroom")

    def _handle_analysis_result(self, result: _HeadroomResult) -> None:
        if self._tracks_table is None:
            return

        if result.row_index < 0 or result.row_index >= self._tracks_table.rowCount():
            return

        table = self._tracks_table
        if result.true_peak_db is not None:
            table.setItem(result.row_index, 3, QTableWidgetItem(f"{result.true_peak_db:.2f}"))
        if result.recommended_gain_db is not None:
            table.setItem(result.row_index, 4, QTableWidgetItem(f"{result.recommended_gain_db:.2f}"))
        if result.status:
            table.setItem(result.row_index, 5, QTableWidgetItem(result.status))

        if result.error_message:
            self._log_message(f"Ошибка анализа {table.item(result.row_index, 0).text()}: {result.error_message}")

        self._analysis_completed += 1
        self._update_progress_bar()

    def _handle_analysis_finished(self) -> None:
        if self.progress_bar is not None and self._analysis_total > 0:
            self.progress_bar.setValue(100)
        if self._analyze_button is not None:
            self._analyze_button.setEnabled(True)
        self._analysis_worker = None
        if self._analysis_thread is not None:
            self._analysis_thread.quit()
            self._analysis_thread.wait()
        self._analysis_thread = None
        self._analysis_total = 0
        self._analysis_completed = 0
        self._log_message("Анализ headroom завершён")

    def _set_rows_status(self, tasks: Sequence[_HeadroomTask], status: str) -> None:
        if self._tracks_table is None:
            return
        for task in tasks:
            self._tracks_table.setItem(task.row_index, 5, QTableWidgetItem(status))
            if status == "analyzing":
                self._tracks_table.setItem(task.row_index, 3, QTableWidgetItem(""))
                self._tracks_table.setItem(task.row_index, 4, QTableWidgetItem(""))

    def _update_progress_bar(self) -> None:
        if self.progress_bar is None:
            return
        if self._analysis_total <= 0:
            self.progress_bar.setValue(0)
            return
        percentage = int((self._analysis_completed / self._analysis_total) * 100)
        self.progress_bar.setValue(max(0, min(100, percentage)))
