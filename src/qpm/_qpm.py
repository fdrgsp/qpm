from __future__ import annotations
from pathlib import Path
import logging

import matplotlib.pyplot as plt
from PyQt6.QtWidgets import (
    QWidget,
    QTabWidget,
    QVBoxLayout,
    QPushButton,
    QHBoxLayout,
    QLineEdit,
    QLabel,
    QCheckBox,
    QProgressBar,
)
from PyQt6.QtCore import Qt
import numpy as np
import pandas as pd
import skimage
from typing import Generator

from ._util import (
    BrowseWidget,
    QPMSettingsDoubleSpinBox,
    QPMSettingsSpinBox,
    create_divider_line,
    show_error_dialog,
)
from ._dpc_algorithm import DPCSolver
from superqt import QIconifyIcon
import tifffile
from cellpose import io
from superqt.utils import create_worker, GeneratorWorker, FunctionWorker
import traceback
from ._segmentation import CellposeSAMSegmentation

# Setup logger
logger = logging.getLogger(__name__)

RED = "#C33"
GREEN = "#04CD04"
QPM_PROCESSED = "_qpm_processed"
PHC_PROCESSED = "_phc_segmented"

BAR_STYLESHEET = """
    QProgressBar {
        border: 1px solid grey;
        border-radius: 5px;
        text-align: center;
    }
    QProgressBar::chunk {
        background-color: #00FF00;
        border-radius: 3px;
    }
"""

TEST_DATA = Path(__file__).parent.parent.parent / "tests" / "_test_data"


class QPMWidget(QWidget):
    """The QPM widget."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)

        self.resize(500, 600)

        self.setWindowTitle("QPM Widget")

        self._worker: FunctionWorker | GeneratorWorker | None = None
        self._cancel_requested: bool = False

        self._dpc_solver: DPCSolver | None = None

        self._skip_files: list[str] = []

        # COMMON WIDGETS------------------------------------------------------------
        # input and output directories
        self._input_dir = BrowseWidget(
            self,
            label="Input Directory",
            tooltip="Directory containing input files.",
            is_dir=True,
        )
        self._output_dir = BrowseWidget(
            self,
            label="Output Directory",
            tooltip="Directory to save output files.",
            is_dir=True,
        )

        # SEGMENTATION WIDGETS (common for both) -----------------------------------
        self._cp_wdg = CellposeSAMSegmentation(self)

        # TABS WIDGET --------------------------------------------------------------
        self._tabwidget = QTabWidget()

        self._qpm_widget = QWidget()
        self._tabwidget.addTab(self._qpm_widget, "QPM")

        self._phc_widget = QWidget()
        self._tabwidget.addTab(self._phc_widget, "Phase Contrast Segmentation")
        lbl = QLabel("Run CellposeSAM segmentation on phase contrast images.")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        phc_layout = QVBoxLayout(self._phc_widget)
        phc_layout.addWidget(lbl)
        phc_layout.addStretch()

        # QPM WIDGET ---------------------------------------------------------------
        # qpm settings
        self._wav = QPMSettingsDoubleSpinBox("Wavelength (µm):", parent=self)
        self._wav.setDecimals(3)
        self._wav.setValue(0.530)

        self._mag = QPMSettingsDoubleSpinBox("Magnification:", parent=self)
        self._mag.setValue(40)

        self._na = QPMSettingsDoubleSpinBox("Numerical Aperture:", parent=self)
        self._na.setValue(0.75)

        self._na_in = QPMSettingsDoubleSpinBox("Numerical Aperture (In):", parent=self)
        self._na_in.setValue(0.0)

        self._cam_pixel_size = QPMSettingsDoubleSpinBox(
            "Camera Pixel Size (µm):", parent=self
        )
        self._cam_pixel_size.setValue(6.5)

        rotation_wdg = QWidget(self)
        rotation_wdg.setToolTip(
            "Rotation angles for each illumination channel, in degrees.\n"
            "Comma-separated values, e.g., '180, 90, 270, 0' for 4 angles."
        )
        r_lbl = QLabel("Rotation (deg):", self)
        self._rotation = QLineEdit(self)
        self._rotation.setText("180, 90, 270, 0")  # default for 4 angles
        rotation_wdg_layout = QHBoxLayout(rotation_wdg)
        rotation_wdg_layout.setContentsMargins(0, 0, 0, 0)
        rotation_wdg_layout.setSpacing(5)
        rotation_wdg_layout.addWidget(r_lbl)
        rotation_wdg_layout.addWidget(self._rotation)

        invert_ph = QWidget(self)
        invert_ph.setToolTip("Invert the phase sign after reconstruction.")
        i_lbl = QLabel("Invert Phase:", self)
        self._invert_ph = QCheckBox(self)
        self._invert_ph.setChecked(False)
        invert_ph_layout = QHBoxLayout(invert_ph)
        invert_ph_layout.setContentsMargins(0, 0, 0, 0)
        invert_ph_layout.setSpacing(5)
        invert_ph_layout.addWidget(i_lbl)
        invert_ph_layout.addWidget(self._invert_ph)

        # tikhonov settings
        self._tikhonov_abs = QPMSettingsDoubleSpinBox("Tikhonov reg_u:", parent=self)
        self._tikhonov_abs.setDecimals(4)
        self._tikhonov_abs.setValue(0.1)
        self._tikhonov_ph = QPMSettingsDoubleSpinBox("Tikhonov reg_p:", parent=self)
        self._tikhonov_ph.setDecimals(4)
        self._tikhonov_ph.setValue(0.005)

        # analysis (QPM)
        analysis_wdg = QWidget(self)
        a_lbl = QLabel("Analysis to perform:", self)
        self._segment_cbox = QCheckBox("Segmentation", self)
        self._segment_cbox.setChecked(True)
        self._qpm_reconstruct_cbox = QCheckBox("QPM Reconstruction", self)
        self._qpm_reconstruct_cbox.setChecked(True)
        analysis_layout = QHBoxLayout(analysis_wdg)
        analysis_layout.setContentsMargins(0, 0, 0, 0)
        analysis_layout.setSpacing(10)
        analysis_layout.addWidget(a_lbl)
        analysis_layout.addWidget(self._qpm_reconstruct_cbox)
        analysis_layout.addWidget(self._segment_cbox)
        analysis_layout.addStretch()

        # BOTTOM WIDGET -------------------------------------------------------------
        self._progress_bar = QProgressBar(self)
        self._progress_bar.setTextVisible(True)
        self._progress_bar.setStyleSheet(BAR_STYLESHEET)
        self._progress_bar.setFixedHeight(15)

        # buttons
        self._run_btn = QPushButton("Run QPM Processing")
        self._run_btn.setIcon(QIconifyIcon("famicons:rocket-sharp", color=GREEN))
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setIcon(QIconifyIcon("icomoon-free:stop", color=RED))
        btns_layout = QHBoxLayout()
        btns_layout.setContentsMargins(0, 0, 0, 0)
        btns_layout.setSpacing(5)
        btns_layout.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        btns_layout.addWidget(self._run_btn)
        btns_layout.addWidget(self._cancel_btn)
        btns_layout.addWidget(self._progress_bar)
        self._run_btn.clicked.connect(self.run)
        self._cancel_btn.clicked.connect(self.cancel)

        # LABELS STYLING -------------------------------------------------------------
        dir_size = self._output_dir._label.sizeHint().width()
        self._input_dir._label.setFixedWidth(dir_size)

        fixed_w = self._na_in._label.sizeHint().width()
        for widget in [
            self._wav,
            self._mag,
            self._na,
            self._na_in,
            self._cam_pixel_size,
            self._tikhonov_abs,
            self._tikhonov_ph,
        ]:
            assert isinstance(widget, QPMSettingsDoubleSpinBox) or isinstance(
                widget, QPMSettingsSpinBox
            )
            widget._label.setFixedWidth(fixed_w)
        a_lbl.setFixedWidth(fixed_w)
        r_lbl.setFixedWidth(fixed_w)
        i_lbl.setFixedWidth(fixed_w)

        # LAYOUTS---------------------------------------------------------------------

        # main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        main_layout.addWidget(create_divider_line("Input/Output Directories"))
        main_layout.addWidget(self._input_dir)
        main_layout.addWidget(self._output_dir)

        # CellposeSAM
        main_layout.addWidget(create_divider_line("CellposeSAM Settings"))
        main_layout.addWidget(self._cp_wdg)

        # tabwidget
        main_layout.addWidget(self._tabwidget)

        # qpm layout tab
        qpm_layout = QVBoxLayout(self._qpm_widget)
        qpm_layout.addWidget(create_divider_line("QPM Settings"))
        qpm_layout.addWidget(self._wav)
        qpm_layout.addWidget(self._mag)
        qpm_layout.addWidget(self._na)
        qpm_layout.addWidget(self._na_in)
        qpm_layout.addWidget(self._cam_pixel_size)
        qpm_layout.addWidget(rotation_wdg)
        qpm_layout.addWidget(invert_ph)
        qpm_layout.addWidget(create_divider_line("QPM Tikhonov Settings"))
        qpm_layout.addWidget(self._tikhonov_abs)
        qpm_layout.addWidget(self._tikhonov_ph)
        qpm_layout.addWidget(create_divider_line("Analysis"))
        qpm_layout.addWidget(analysis_wdg)
        qpm_layout.addStretch()

        # buttons layout
        main_layout.addLayout(btns_layout)

        # TO REMOVE, JUST FOR TESTING
        input = TEST_DATA / "input_qpm"
        output = TEST_DATA / "output"
        self._input_dir.setValue(input)
        self._output_dir.setValue(output)
        # END TO REMOVE

        # connections
        self._tabwidget.currentChanged.connect(self._rename_run_buttons)

    # PUBLIC METHODS-------------------------------------------------------------

    def cancel(self) -> None:
        """Cancel the QPM processing."""
        if self._worker is None or not self._worker.is_running:
            return
        self._cancel_requested = True
        self._worker.quit()
        # clear the progress bar
        self._progress_bar.setValue(0)
        self._progress_bar.setFormat("")

    def run(self) -> None:
        """Run the QPM processing in a separate thread."""
        self._cancel_requested = False
        self._skip_files.clear()

        self._enable(False)

        # qpm tab
        if self._tabwidget.currentIndex() == 0:
            logger.info("STARTING QPM PROCESSING")
            self._worker = create_worker(
                self._run_qpm,
                _start_thread=True,
                _connect={
                    "yielded": self._update_progress,
                    "finished": self._on_processing_finished,
                    "errored": self._on_error,
                },
            )

        # phase contrast tab
        if self._tabwidget.currentIndex() == 1:
            logger.info("STARTING PHASE CONTRAST SEGMENTATION")
            self._worker = create_worker(
                self._run_phc,
                _start_thread=True,
                _connect={
                    "yielded": self._update_progress,
                    "finished": self._on_processing_finished,
                    "errored": self._on_error,
                },
            )

    # COMMON PRIVATE METHODS-------------------------------------------------------------

    def _enable(self, enable: bool) -> None:
        """Enable or disable all input widgets."""
        for widget in [
            self._input_dir,
            self._output_dir,
            self._tabwidget,
            self._cp_wdg,
        ]:
            widget.setEnabled(enable)

    def _rename_run_buttons(self) -> None:
        """Rename the run buttons based on the selected tab."""
        if self._tabwidget.currentIndex() == 0:
            self._run_btn.setText("Run QPM Processing")
        elif self._tabwidget.currentIndex() == 1:
            self._run_btn.setText("Run PhC Segmentation")

    def _update_progress(self, progress_data: dict) -> None:
        """Update the progress bar."""
        if progress_data["type"] == "init":
            self._progress_bar.setMaximum(progress_data["total"])
            self._progress_bar.setValue(0)
            self._progress_bar.setFormat(f"0/{progress_data['total']}")
            self._progress_bar.setVisible(True)
        elif progress_data["type"] == "update":
            current = progress_data["current"]
            total = self._progress_bar.maximum()
            self._progress_bar.setValue(current)
            self._progress_bar.setFormat(f"{current}/{total}")
        elif progress_data["type"] == "error":
            show_error_dialog(self, progress_data["message"])
        elif progress_data["type"] == "validation_errors":
            show_error_dialog(self, progress_data["message"])

    def _on_processing_finished(self) -> None:
        """Called when processing is finished."""
        self._cancel_requested = False
        logger.info("PROCESSING COMPLETED SUCCESSFULLY")
        self._enable(True)

    def _on_error(self, exc: Exception) -> None:
        logger.error(f"Processing failed: {exc}")
        traceback.print_exception(type(exc), exc, exc.__traceback__)
        self._cancel_requested = False
        self._progress_bar.setValue(0)
        self._progress_bar.setFormat("")
        self._enable(True)

    def _get_sorted_files_and_dirs(self) -> tuple[list[Path], int]:
        """Get sorted list of files and directories to process, plus total file count.

        Returns:
            tuple: (sorted_items, total_files) where sorted_items contains both
                   files and directories sorted by name, and total_files is the
                   count of .tif/.tiff files that will be processed.
        """
        path = Path(self._input_dir.value())
        items = sorted(path.iterdir(), key=lambda x: x.name)
        total_files = 0

        for item in items:
            if item.is_file() and item.suffix in {".tif", ".tiff"}:
                total_files += 1
            elif item.is_dir():
                # Count both .tif and .tiff files in subdirectories
                total_files += len(list(item.glob("*.tif"))) + len(
                    list(item.glob("*.tiff"))
                )

        return items, total_files

    def _segment_file(
        self, image: np.ndarray | None, name: str, output_dir: Path
    ) -> np.ndarray | None:
        """Process a single TIF file."""
        if image is None:
            return None

        logger.info(f"Processing {name}...")

        if self._cancel_requested:
            return None

        # run segmentation
        logger.info("Running CellposeSAM segmentation...")

        labels, _, _ = self._cp_wdg.segment(image)

        # save the labels
        logger.info("Saving labels...")
        io.imsave(output_dir / f"{name}_labels.tif", labels)
        # This I would remove later on or make it optional.
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        ax[0].imshow(image, cmap="gray")
        ax[0].set_title(
            "QPM Image"
            if self._tabwidget.currentIndex() == 0
            else "Phase Contrast Image"
        )
        ax[0].axis("off")
        ax[1].imshow(labels, cmap="nipy_spectral")
        ax[1].set_title("Labels")
        ax[1].axis("off")
        plt.tight_layout()
        plt.savefig(output_dir / f"{name}_labels.png", dpi=150)
        plt.close()
        logger.info(f"Saved labels for {name}")
        return labels

    # QPM PRIVATE METHODS-------------------------------------------------------

    def _run_qpm(self) -> Generator[dict, None, None]:
        """Run the QPM processing."""
        if not self._input_dir.value():
            return

        if not self._output_dir.value():
            yield {"type": "error", "message": "Output directory is not set."}
            return

        rotations = self._parse_rotation()
        if not rotations:
            yield {"type": "error", "message": "Invalid rotation angles format."}
            return

        sorted_items, num_files = self._get_sorted_files_and_dirs()
        failed_files = []  # Collect failed files

        # Initialize progress bar
        yield {"type": "init", "total": num_files}

        # understand if the solver should be initialized at each image or not
        # self._dpc_solver = None

        current_file = 0

        for item in sorted_items:
            if self._cancel_requested:
                return

            if item.is_file() and item.suffix in {".tif", ".tiff"}:
                # Process and yield progress updates in real-time
                for progress_update in self._process_qpm_tif_generator(item, rotations):
                    if progress_update["type"] == "progress":
                        current_file += 1
                        yield {"type": "update", "current": current_file}
                    elif progress_update["type"] == "error":
                        failed_files.append(progress_update["message"])

            elif item.is_dir():
                tif_files = list(item.glob("*.tif")) + list(item.glob("*.tiff"))
                for tif_file in sorted(tif_files, key=lambda x: x.name):
                    # Process and yield progress updates in real-time
                    for progress_update in self._process_qpm_tif_generator(
                        tif_file, rotations
                    ):
                        if progress_update["type"] == "progress":
                            current_file += 1
                            yield {"type": "update", "current": current_file}
                        elif progress_update["type"] == "error":
                            failed_files.append(progress_update["message"])

        # Report failed files at the end
        if failed_files:
            error_message = "The following files failed validation:\n\n" + "\n".join(
                failed_files
            )
            yield {"type": "validation_errors", "message": error_message}

    def _process_qpm_tif_generator(
        self, tif_path: Path, rotations: list[float]
    ) -> Generator[dict, None, None]:
        """Process a single .tif file for QPM with real-time progress updates.

        Yields progress updates for each position processed.
        """
        name = tif_path.stem.replace(".ome", "")

        with tifffile.TiffFile(tif_path) as tif:
            positions = tif.series

            if not positions:
                # simple TIFF files (no OME metadata)
                image = tifffile.imread(tif_path)
                ok, msg = self._process_single_qpm_image(image, name, rotations)
                if not ok:
                    yield {"type": "error", "message": msg}
                else:
                    yield {"type": "progress"}

            elif len(positions) == 1:
                # OME-TIFF with single position
                image = positions[0].asarray()
                ok, msg = self._process_single_qpm_image(image, name, rotations)
                if not ok:
                    yield {"type": "error", "message": msg}
                else:
                    yield {"type": "progress"}

            else:
                # OME-TIFF with multiple positions - yield progress for each position
                for pos in positions:
                    if self._cancel_requested:
                        break

                    self._skip_files.append(pos.name)
                    image = pos.asarray()

                    ok, msg = self._process_single_qpm_image(image, pos.name, rotations)
                    if not ok:
                        yield {"type": "error", "message": msg}
                        break
                    else:
                        yield {"type": "progress"}

    def _process_single_qpm_image(
        self, image: np.ndarray, name: str, rotations: list[float]
    ) -> tuple[bool, str]:
        """Process a single QPM image.

        Args:
            image: The image array to process
            name: The name to use for output files
            rotations: The rotation angles for QPM reconstruction

        Returns:
            (False, "{name}: {error}") on validation failure or (True, "") on success.
        """
        is_valid, error_msg = self._validate_qpm_image(image)
        if not is_valid:
            return False, f"{name}: {error_msg}"

        out = Path(self._output_dir.value()) / f"{name}{QPM_PROCESSED}"
        out.mkdir(parents=True, exist_ok=True)

        ph, seg = None, None
        if self._qpm_reconstruct_cbox.isChecked():
            self._dpc_solver = None
            ph = self._reconstruct_qpm(image, name, rotations, out)
        if self._segment_cbox.isChecked():
            seg = self._segment_file(ph, name, out)
        if ph is not None and seg is not None:
            self._generate_csv_file(seg, ph, name, out)

        return True, ""

    def _validate_qpm_image(self, image: np.ndarray) -> tuple[bool, str]:
        """Validate the input image.

        Returns:
            tuple: (is_valid, error_message)
        """
        if image.ndim != 3:
            return (
                False,
                f"The file should have 3 dimensions (C, H, W), got {image.ndim}.",
            )
        if image.shape[0] != 4:
            return False, "The file should have 4 channels, one per illumination angle."
        return True, ""

    def _parse_rotation(self) -> list[float]:
        """Parse the rotation angles from the input."""
        try:
            return [float(angle.strip()) for angle in self._rotation.text().split(",")]
        except ValueError:
            show_error_dialog(self, "Invalid rotation angles format.")
            return []

    def _reconstruct_qpm(
        self, image: np.ndarray, name: str, rotations: list[float], output_dir: Path
    ) -> np.ndarray | None:
        """Reconstruct the QPM image.

        Code form Laura Waller Lab: https://github.com/Waller-Lab/DPC/tree/master/python_code
        """
        logger.info(f"Reconstructing QPM for {name}...")

        if self._dpc_solver is None:
            logger.info("Initializing DPCSolver...")
            self._dpc_solver = DPCSolver(
                image,
                self._wav.value(),
                self._na.value(),
                self._na_in.value(),
                self._cam_pixel_size.value() / self._mag.value(),
                rotations,
                dpc_num=4,
            )
        else:
            self._dpc_solver.set_dpc_imgs(image)
            self._dpc_solver.normalization()

        if self._cancel_requested:
            return None

        # solve DPC Deconvoltion Problems
        # parameters for Tikhonov regurlarization [u:absorption, p:phase]
        # (need to tune this based on SNR)
        logger.info("Solving DPC Deconvoltion Problems...")
        self._dpc_solver.setRegularizationParameters(
            reg_u=self._tikhonov_abs.value(), reg_p=self._tikhonov_ph.value()
        )
        dpc_result = self._dpc_solver.solve(method="Tikhonov")

        logger.info("Saving QPM results...")
        abs = dpc_result[0].real.astype("float32")
        ph = dpc_result[0].imag.astype("float32")
        if self._invert_ph.isChecked():
            ph = -ph
        tifffile.imwrite(output_dir / f"frame_{name}_abs.tif", abs, imagej=True)
        tifffile.imwrite(output_dir / f"frame_{name}_ph.tif", ph, imagej=True)
        logger.info(f"Saved QPM results for {name}")
        return ph

    # PHASE CONTRAST PRIVATE METHODS------------------------------------------------

    def _run_phc(self) -> Generator[dict, None, None]:
        """Run the Phase Contrast processing."""
        if not self._input_dir.value():
            return

        if not self._output_dir.value():
            yield {"type": "error", "message": "Output directory is not set."}
            return

        sorted_items, num_files = self._get_sorted_files_and_dirs()
        failed_files = []  # Collect failed files

        # Initialize progress bar
        yield {"type": "init", "total": num_files}

        current_file = 0
        for item in sorted_items:
            if self._cancel_requested:
                return

            if item.is_file() and item.suffix in {".tif", ".tiff"}:
                if item.stem.replace(".ome", "") in self._skip_files:
                    continue
                # process and yield progress updates in real-time
                for progress_update in self._process_phc_tif_generator(item):
                    if progress_update["type"] == "progress":
                        current_file += 1
                        yield {"type": "update", "current": current_file}
                    elif progress_update["type"] == "error":
                        failed_files.append(progress_update["message"])

            elif item.is_dir():
                tif_files = list(item.glob("*.tif")) + list(item.glob("*.tiff"))
                for tif_file in sorted(tif_files, key=lambda x: x.name):
                    if tif_file.stem.replace(".ome", "") in self._skip_files:
                        continue
                    # process and yield progress updates in real-time
                    for progress_update in self._process_phc_tif_generator(tif_file):
                        if progress_update["type"] == "progress":
                            current_file += 1
                            yield {"type": "update", "current": current_file}
                        elif progress_update["type"] == "error":
                            failed_files.append(progress_update["message"])

        # Report failed files at the end
        if failed_files:
            error_message = "The following files failed validation:\n\n" + "\n".join(
                failed_files
            )
            yield {"type": "validation_errors", "message": error_message}

    def _process_phc_tif_generator(self, tif_path: Path) -> Generator[dict, None, None]:
        """Process a single .tif file for Phase Contrast with real-time progress updates.

        Yields progress updates for each position processed.
        """
        name = tif_path.stem.replace(".ome", "")

        with tifffile.TiffFile(tif_path) as tif:
            positions = tif.series

            if not positions:
                # simple TIFF files (no OME metadata)
                image = tifffile.imread(tif_path)
                ok, msg = self._process_single_phc_image(image, name)
                if not ok:
                    yield {"type": "error", "message": msg}
                else:
                    yield {"type": "progress"}

            elif len(positions) == 1:
                # OME-TIFF with single position
                image = positions[0].asarray()
                ok, msg = self._process_single_phc_image(image, name)
                if not ok:
                    yield {"type": "error", "message": msg}
                else:
                    yield {"type": "progress"}

            else:
                # OME-TIFF with multiple positions - yield progress for each position
                for pos in positions:
                    if self._cancel_requested:
                        break

                    self._skip_files.append(pos.name)
                    image = pos.asarray()

                    ok, msg = self._process_single_phc_image(image, pos.name)
                    if not ok:
                        yield {"type": "error", "message": msg}
                        break
                    else:
                        yield {"type": "progress"}

    def _process_single_phc_image(
        self, image: np.ndarray, name: str
    ) -> tuple[bool, str]:
        """Process a single phase contrast image.

        Args:
            image: The image array to process
            name: The name to use for output files

        Returns:
            (False, "{name}: {error}") on validation failure or (True, "") on success.
        """
        is_valid, error_msg = self._validate_phc_image(image)
        if not is_valid:
            return False, f"{name}: {error_msg}"

        out = Path(self._output_dir.value()) / f"{name}{PHC_PROCESSED}"
        out.mkdir(parents=True, exist_ok=True)

        self._segment_file(image, name, out)
        return True, ""

    def _validate_phc_image(self, image: np.ndarray) -> tuple[bool, str]:
        """Validate the input image.

        Returns:
            tuple: (is_valid, error_message)
        """
        if image.ndim != 2:
            return (
                False,
                "The file should have 2 dimensions (H, W) for phase contrast images.",
            )
        return True, ""

    # CSV GENERATION METHODS-------------------------------------------------------

    def _generate_csv_file(
        self, labels: np.ndarray, phase_image: np.ndarray, name: str, output_dir: Path
    ) -> None:
        """Generate a CSV file with measurements."""
        name = name.replace(".ome", "")
        logger.info(f"Generating CSV file for {name}...")

        if self._cancel_requested:
            return None

        props_dict = skimage.measure.regionprops_table(
            labels,
            intensity_image=phase_image,
            properties=[
                "label",
                "intensity_mean",
                "area",
                "eccentricity",
                "axis_major_length",
                "axis_minor_length",
            ],
        )

        # calculate dry mass for each region and add it to the props_dict
        self._extract_and_add_dry_mass_calculation(labels, phase_image, props_dict)

        props_df = pd.DataFrame(props_dict)
        logger.info(f"Saving CSV file for {name}...")
        props_df.to_csv(output_dir / f"{name}_measurements.csv", index=False)
        logger.info(f"Saved CSV file for {name}")

        # Temporary plots for testing
        props_df["dry_mass"].plot(
            kind="hist",
            bins=200,
            title="Dry Mass Distribution",
            xlabel="Dry Mass (pg)",
            ylabel="Frequency",
        )
        plt.savefig(output_dir / f"{name}_dry_mass_distribution.png", dpi=150)
        plt.close()

        props_df["area"].plot(
            kind="hist",
            bins=200,
            title="Area Distribution",
            xlabel="Area (pixels)",
            ylabel="Frequency",
        )
        plt.savefig(output_dir / f"{name}_area_distribution.png", dpi=150)
        plt.close()

        props_df["eccentricity"].plot(
            kind="hist",
            bins=200,
            title="Eccentricity Distribution",
            xlabel="Eccentricity",
            ylabel="Frequency",
        )
        plt.savefig(output_dir / f"{name}_eccentricity_distribution.png", dpi=150)
        plt.close()

        props_df["axis_major_length"].plot(
            kind="hist",
            bins=200,
            title="Axis Major Length Distribution",
            xlabel="Axis Major Length (pixels)",
            ylabel="Frequency",
        )
        plt.savefig(output_dir / f"{name}_axis_major_length_distribution.png", dpi=150)
        plt.close()

    def _extract_and_add_dry_mass_calculation(
        self, labels: np.ndarray, phase_image: np.ndarray, props_dict: dict
    ) -> None:
        """Extract intensity sum and calculate dry mass for each region.

        Based on the formula from the paper:
        - https://pmc.ncbi.nlm.nih.gov/articles/PMC5730079/
        - dry mass M = (λ/(2π)) * (1/α) * (pixel_area) * Σ phase_values
        """
        props = skimage.measure.regionprops(labels, intensity_image=phase_image)
        intensity_sums = []
        for region in props:
            region_mask = region.image
            region_intensity = region.image_intensity
            intensity_sum = np.sum(region_intensity[region_mask])
            intensity_sums.append(intensity_sum)

        props_dict["intensity_sum"] = np.array(intensity_sums)

        # calculate dry mass for each region using the exact formula from the
        # paper https://pmc.ncbi.nlm.nih.gov/articles/PMC5730079/:
        # equation (5): M = (λ/(2π)) * (1/α) * (pixel_area) * Σ phase_values
        wavelength = self._wav.value() * 1e-6  # meters
        alpha = 0.2e-3  # refractive index increment in m³/kg (0.2 mL/g from paper)
        pixel_size_m = (
            self._cam_pixel_size.value() / self._mag.value()
        ) * 1e-6  # meters
        pixel_area_m2 = pixel_size_m**2  # physical area per pixel in m²
        # calculate dry mass factor: (λ/(2π)) * (1/α) * pixel_area * conversion to picograms
        dry_mass_factor = (
            (wavelength / (2 * np.pi)) * (1 / alpha) * pixel_area_m2 * 1e15
        )
        props_dict["dry_mass"] = props_dict["intensity_sum"] * dry_mass_factor
