from __future__ import annotations
from typing import Any
from cellpose import models, core, io
import numpy as np
from PyQt6.QtWidgets import QGroupBox, QWidget, QVBoxLayout
from superqt import QCollapsible
from ._util import QPMSettingsDoubleSpinBox, QPMSettingsSpinBox


class CellposeSAMSegmentation(QGroupBox):
    """A class to handle cellpose segmentation of bacteria images."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent=parent)

        io.logger_setup()

        self._model = models.CellposeModel(gpu=core.use_gpu(), pretrained_model="cpsam")

        # GUI ELEMENTS ---------------------------------------------------------------
        self._cellpose_wdg = QCollapsible("Cellpose Settings", parent=self)

        # diameter
        self._diameter = QPMSettingsSpinBox("Diameter in px:", parent=self)
        self._diameter.setSpecialValueText("Auto")
        self._diameter.setToolTip(
            "Cellpose rescales the image so objects match this diameter before "
            "segmentation (model trained for ~30 px). If set to 'Auto' (or 0, the "
            "default), Cellpose  will estimate it."
        )

        # flow threshold
        self._flow_threshold = QPMSettingsDoubleSpinBox("Flow Threshold:", parent=self)
        self._flow_threshold.setDecimals(2)
        self._flow_threshold.setValue(0.4)
        self._flow_threshold.setToolTip(
            "Flow error threshold, all cells with errors below threshold are kept."
            "(not used for 3D). Defaults to 0.4"
        )

        # cellprob threshold
        self._cellprob_threshold = QPMSettingsDoubleSpinBox(
            "Cellprob Threshold:", parent=self
        )
        self._cellprob_threshold.setDecimals(2)
        self._cellprob_threshold.setValue(0.0)
        self._cellprob_threshold.setToolTip(
            "All pixels with value above threshold kept for masks, decrease to find "
            "more and larger masks. Defaults to 0.0."
        )

        # min size
        self._min_size = QPMSettingsSpinBox("Min Size (pixels):", parent=self)
        self._min_size.setValue(15)
        self._min_size.setToolTip(
            "All ROIs below this size, in pixels, will be discarded. Defaults to 15."
        )

        # max size fraction
        self._max_size_fraction = QPMSettingsDoubleSpinBox(
            "Max Size Fraction:", parent=self
        )
        self._max_size_fraction.setDecimals(2)
        self._max_size_fraction.setValue(0.4)
        self._max_size_fraction.setToolTip(
            "Masks larger than max_size_fraction of total image size are removed. "
            "Default is 0.4."
        )

        # tile norm blocksize
        self._tile_norm_blocksize = QPMSettingsSpinBox(
            "Tile Norm Blocksize:", parent=self
        )
        self._tile_norm_blocksize.setValue(0)
        self._tile_norm_blocksize.setToolTip(
            "Block size for tile normalization. Defaults to 0."
        )

        # batch size
        self._batch_size = QPMSettingsSpinBox("Batch Size:", parent=self)
        self._batch_size.setValue(8)
        self._batch_size.setToolTip(
            "Number of images to process in a batch. "
            "Increase if you have more memory. Defaults to 8."
        )

        # labels style
        fixed = self._tile_norm_blocksize._label.sizeHint().width()
        for wdg in (
            self._flow_threshold,
            self._cellprob_threshold,
            self._min_size,
            self._max_size_fraction,
            self._tile_norm_blocksize,
            self._batch_size,
            self._diameter,
        ):
            wdg._label.setFixedWidth(fixed)

        # set collapsible content
        content = QWidget()
        cellpose_layout = QVBoxLayout(content)
        cellpose_layout.setContentsMargins(0, 0, 0, 0)
        cellpose_layout.setSpacing(5)
        cellpose_layout.addWidget(self._flow_threshold)
        cellpose_layout.addWidget(self._cellprob_threshold)
        cellpose_layout.addWidget(self._min_size)
        cellpose_layout.addWidget(self._max_size_fraction)
        cellpose_layout.addWidget(self._tile_norm_blocksize)
        cellpose_layout.addWidget(self._batch_size)
        cellpose_layout.addWidget(self._diameter)
        self._cellpose_wdg.setContent(content)

        # main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        main_layout.addWidget(self._cellpose_wdg)

        # collapse by default
        self.collapse()

    def collapse(self) -> None:
        """Collapse the settings panel."""
        self._cellpose_wdg.collapse()

    def value(self) -> dict[str, Any]:
        """Get current parameters as a dictionary."""
        return {
            "flow_threshold": self._flow_threshold.value(),
            "cellprob_threshold": self._cellprob_threshold.value(),
            "min_size": self._min_size.value(),
            "max_size_fraction": self._max_size_fraction.value(),
            "tile_norm_blocksize": self._tile_norm_blocksize.value(),
            "batch_size": self._batch_size.value(),
            "diameter": self._diameter.value() or None,
        }

    def segment(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run CellposeSAM segmentation on the given image."""
        values = self.value()
        values.pop("tile_norm_blocksize")
        normalize = {"tile_norm_blocksize": self._tile_norm_blocksize.value()}
        labels, flows, styles = self._model.eval(image, **values, normalize=normalize)
        return labels, flows, styles
