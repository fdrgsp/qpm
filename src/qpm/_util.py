from __future__ import annotations
from pathlib import Path
from PyQt6.QtWidgets import (
    QWidget,
    QLineEdit,
    QPushButton,
    QFileDialog,
    QLabel,
    QSizePolicy,
    QHBoxLayout,
    QMessageBox,
    QDoubleSpinBox,
    QFrame,
)


def show_error_dialog(parent: QWidget, message: str) -> None:
    """Show an error dialog with the given message."""
    dialog = QMessageBox(parent)
    dialog.setWindowTitle("Error")
    dialog.setText(message)
    dialog.setIcon(QMessageBox.Icon.Critical)
    dialog.setStandardButtons(QMessageBox.StandardButton.Ok)
    dialog.exec()


class BrowseWidget(QWidget):
    """A widget that allows browsing for a file or directory."""

    def __init__(
        self,
        parent: QWidget | None = None,
        label: str = "",
        path: str | None = None,
        tooltip: str = "",
        *,
        is_dir: bool = True,
    ) -> None:
        super().__init__(parent)

        self._is_dir = is_dir

        self._label_text = label

        self._label = QLabel(f"{self._label_text}:")
        self._label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._label.setToolTip(tooltip)
        self._path = QLineEdit()
        self._path.setText(path or "")
        self._browse_btn = QPushButton("Browse")
        self._browse_btn.clicked.connect(self._on_browse)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        layout.addWidget(self._label)
        layout.addWidget(self._path)
        layout.addWidget(self._browse_btn)

    def value(self) -> str:
        import os

        path_text = self._path.text()
        return str(os.path.normpath(path_text)) if path_text else ""

    def setValue(self, path: str | Path) -> None:
        self._path.setText(str(path))

    def _on_browse(self) -> None:
        if not self._is_dir:
            show_error_dialog(
                self, "This widget is configured to browse directories only."
            )
            return
        if path := QFileDialog.getExistingDirectory(
            self, f"Select the {self._label_text}.", self.value()
        ):
            self._path.setText(path)


class QPMSettingsSpinBox(QWidget):
    """A spin box widget for QPM settings."""

    def __init__(self, label: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._label = QLabel(label)
        self._label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._spin = QDoubleSpinBox()
        self._spin.setRange(0.0, 100000.0)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        layout.addWidget(self._label)
        layout.addWidget(self._spin)

    def value(self) -> float:
        """Get the current value of the spin box."""
        return self._spin.value()

    def setValue(self, value: float) -> None:
        """Set the value of the spin box."""
        self._spin.setValue(value)

    def setDecimals(self, decimals: int) -> None:
        """Set the number of decimal places for the spin box."""
        self._spin.setDecimals(decimals)

    def setReadOnly(self, read_only: bool) -> None:
        """Set the read-only state of the spin box."""
        self._spin.setReadOnly(read_only)

    def setSpecialValueText(self, text: str) -> None:
        """Set the special value text for the spin box."""
        self._spin.setSpecialValueText(text)


def create_divider_line(text: str | None = None) -> QWidget:
    """Create a horizontal divider line, optionally with text.

    Parameters
    ----------
    text : str | None
        Optional text to display in front of the divider line

    Returns
    -------
    QWidget
        Widget containing the divider line and optional text
    """
    if text is None:
        return _create_line()
    # Create container widget for text + line
    container = QWidget()
    layout = QHBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(10)

    # Add text label
    label = QLabel(text)
    # make bold and increase font size
    # label.setStyleSheet("font-weight: bold; font-size: 14px; color: rgb(0, 183, 0);")
    label.setStyleSheet("font-weight: bold; font-size: 14px;")
    layout.addWidget(label)

    line = _create_line()
    layout.addWidget(line, 1)  # Give line stretch factor of 1

    return container


def _create_line() -> QFrame:
    """Create a horizontal line frame for use as a divider."""
    result = QFrame()
    # set color
    # result.setStyleSheet("color: rgb(0, 183, 0);")
    result.setFrameShape(QFrame.Shape.HLine)
    result.setFrameShadow(QFrame.Shadow.Plain)
    return result
