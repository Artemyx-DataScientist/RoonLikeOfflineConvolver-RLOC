"""Main window for the RLOC GUI."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QFileDialog,
    QPushButton,
    QProgressBar,
    QRadioButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtWidgets import QHeaderView


class MainWindow(QMainWindow):
    """Main application window with controls and placeholders."""

    _AUDIO_EXTENSIONS: tuple[str, ...] = (".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav")

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("RLOC GUI")
        self._log_view: QTextEdit | None = None
        self.progress_bar: QProgressBar | None = None
        self._tracks_table: QTableWidget | None = None
        self._known_files: set[str] = set()
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

        settings_layout.addWidget(source_group)
        settings_layout.addWidget(filter_group)
        settings_layout.addWidget(parameters_group)
        settings_layout.addWidget(ear_gain_group)
        settings_layout.addWidget(output_group)
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

        filter_input = QLineEdit()
        filter_input.setPlaceholderText("filter.zip")

        browse_button = QPushButton("Обзор")
        browse_button.clicked.connect(lambda: self._log_message("TODO: выбрать фильтр"))

        layout.addWidget(filter_input)
        layout.addWidget(browse_button)

        return group

    def _create_parameters_group(self) -> QGroupBox:
        group = QGroupBox("Параметры")
        layout = QVBoxLayout(group)

        target_tp_layout = QHBoxLayout()
        target_label = QLabel("target_tp")
        target_tp_spin = QDoubleSpinBox()
        target_tp_spin.setRange(-60.0, 6.0)
        target_tp_spin.setSingleStep(0.1)
        target_tp_spin.setValue(-0.1)
        target_tp_layout.addWidget(target_label)
        target_tp_layout.addWidget(target_tp_spin)

        oversample_layout = QHBoxLayout()
        oversample_label = QLabel("oversample")
        oversample_combo = QComboBox()
        oversample_combo.addItems(["4", "8"])
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
