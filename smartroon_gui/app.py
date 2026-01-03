"""Application entry logic for the RLOC GUI."""

from __future__ import annotations

import sys
from typing import Iterable

from PySide6.QtWidgets import QApplication

from smartroon_gui.main_window import MainWindow


def _create_application(argv: Iterable[str] | None = None) -> QApplication:
    """Create or return an existing :class:`QApplication` instance."""
    existing_app = QApplication.instance()
    if existing_app is not None:
        return existing_app

    return QApplication(list(argv) if argv is not None else sys.argv)


def run(argv: Iterable[str] | None = None) -> int:
    """Initialize and run the GUI application."""
    app = _create_application(argv)
    window = MainWindow()
    window.show()
    return app.exec()
