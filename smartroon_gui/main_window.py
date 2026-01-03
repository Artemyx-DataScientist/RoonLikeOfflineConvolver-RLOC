"""Main window for the RLOC GUI."""

from __future__ import annotations

from PySide6.QtWidgets import QMainWindow


class MainWindow(QMainWindow):
    """Empty main window with a predefined title."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("RLOC GUI")
