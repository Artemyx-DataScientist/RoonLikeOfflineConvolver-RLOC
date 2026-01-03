"""Main window for the RLOC GUI."""

from __future__ import annotations

from pathlib import Path

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


class MainWindow(QMainWindow):
    """Main application window with controls and placeholders."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("RLOC GUI")
        self.filter_input: QLineEdit | None = None
        self.sample_rate_dropdown: QComboBox | None = None
        self.filter_info_label: QLabel | None = None
        self._log_view: QTextEdit | None = None
        self.progress_bar: QProgressBar | None = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        central_widget = QWidget(self)
        main_layout = QVBoxLayout(central_widget)

        content_layout = QHBoxLayout()
        settings_panel = self._create_settings_panel()
        tracks_table = self._create_tracks_table()
        content_layout.addWidget(settings_panel, 0)
        content_layout.addWidget(tracks_table, 1)
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
        select_files_button.clicked.connect(lambda: self._log_message("TODO: выбрать файлы"))

        select_folder_button = QPushButton("Выбрать папку")
        select_folder_button.clicked.connect(lambda: self._log_message("TODO: выбрать папку"))

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
