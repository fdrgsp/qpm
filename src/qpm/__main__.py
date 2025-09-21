from __future__ import annotations
import logging
import sys
import traceback
from types import TracebackType

from PyQt6.QtWidgets import QApplication
from qpm._qpm import QPMWidget


def _our_excepthook(
    type: type[BaseException], value: BaseException, tb: TracebackType | None
) -> None:
    """Excepthook that prints the traceback to the console.

    By default, Qt's excepthook raises sys.exit(), which is not what we want.
    """
    # this could be elaborated to do all kinds of things...
    traceback.print_exception(type, value, tb)


def main() -> None:
    """Main function to run the QPM widget."""
    # Set up logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    app = QApplication(sys.argv)
    win = QPMWidget()
    win.show()
    sys.excepthook = _our_excepthook
    app.exec()


if __name__ == "__main__":
    main()
