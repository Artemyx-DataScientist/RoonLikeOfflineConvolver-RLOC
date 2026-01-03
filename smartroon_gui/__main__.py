"""Module entrypoint for ``python -m smartroon_gui``."""

from __future__ import annotations

from smartroon_gui.app import run


def main() -> int:
    """Execute the GUI application."""
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
