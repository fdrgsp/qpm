from __future__ import annotations
from pathlib import Path
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
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt
import numpy as np
import pandas as pd
import skimage
from typing import Generator
from ._util import (
    BrowseWidget,
    QPMSettingsSpinBox,
    create_divider_line,
    show_error_dialog,
)
from ._dpc_algorithm import DPCSolver
from superqt import QIconifyIcon
from ._segmentation import CellposeSAMSegmentation
import tifffile
from cellpose import core, io, models
from superqt.utils import create_worker, GeneratorWorker, FunctionWorker

RED = "#C33"
GREEN = "#7300FF"
PROCESSED = "_processed"

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

TEST_DATA = Path(__file__).parent / "_test_data"
CAT_PATH = Path(__file__).parent / "_todisplay"


class QPMWidget(QWidget):
    """The QPM widget."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)

        self.setWindowTitle("QPM Widget")
        self.resize(600, 400)

        self._worker: FunctionWorker | GeneratorWorker | None = None
        self._cancel_requested: bool = False

        self._dpc_solver: DPCSolver | None = None

        # segmentation
        self._cp = CellposeSAMSegmentation()
        self._cp2 = models.CellposeModel(gpu=core.use_gpu(), pretrained_model="cpsam")

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

        self._tabs = QTabWidget()  # just an instance of the tab widget
        self.qpm_widget = QWidget()  # where all the QPM-related widgets will go
        self._tabs.addTab(
            self.qpm_widget, "QPM"
        )  # physically add the qpm_widget to the tab widget

        self.phc_widget = QWidget()  # where all the PHC-related widgets will go
        self._tabs.addTab(
            self.phc_widget, "General segmentation"
        )  # physically add the tab

        # qpm settings
        self._wav = QPMSettingsSpinBox("Wavelength (µm):", parent=self)
        self._wav.setDecimals(3)
        self._wav.setValue(0.530)

        self._mag = QPMSettingsSpinBox("Magnification:", parent=self)
        self._mag.setValue(40)

        self._na = QPMSettingsSpinBox("Numerical Aperture:", parent=self)
        self._na.setValue(0.75)

        self._na_in = QPMSettingsSpinBox("Numerical Aperture (In):", parent=self)
        self._na_in.setValue(0.0)

        self._cam_pixel_size = QPMSettingsSpinBox(
            "Camera Pixel Size (µm):", parent=self
        )
        self._cam_pixel_size.setValue(6.5)

        # self._num_channels = QPMSettingsSpinBox("Number of Channels:", parent=self)
        # self._num_channels.setDecimals(0)
        # self._num_channels.setValue(4)

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
        self._tikhonov_abs = QPMSettingsSpinBox("Tikhonov reg_u:", parent=self)
        self._tikhonov_abs.setDecimals(4)
        self._tikhonov_abs.setValue(0.1)
        self._tikhonov_ph = QPMSettingsSpinBox("Tikhonov reg_p:", parent=self)
        self._tikhonov_ph.setDecimals(4)
        self._tikhonov_ph.setValue(0.005)

        # label styling
        fixed_w = self._na_in._label.sizeHint().width()
        for widget in [
            self._wav,
            self._mag,
            self._na,
            self._na_in,
            self._cam_pixel_size,
            # self._num_channels,
            self._tikhonov_abs,
            self._tikhonov_ph,
        ]:
            widget._label.setFixedWidth(fixed_w)
        self._input_dir._label.setFixedWidth(fixed_w)
        self._output_dir._label.setFixedWidth(fixed_w)
        r_lbl.setFixedWidth(fixed_w)
        i_lbl.setFixedWidth(fixed_w)

        # Phase settings -- diameter
        self._use_diam = QCheckBox("Use Diameter for Cellpose", parent=self)
        self._use_diam.setChecked(False)
        self._use_diam.setToolTip(
            "Diameters are used to rescale the image to 30 pix cell diameter."
        )

        self._diameter = QPMSettingsSpinBox("Diameter", parent=self)
        self._diameter.setDecimals(0)
        self._diameter.setSpecialValueText("—")  # shows a dash when at minimum
        self._diameter.setToolTip(
            "Diameters are used to rescale the image to 30 pix cell diameter."
        )

        # React to user toggling the checkbox
        self._use_diam.toggled.connect(self._on_use_toggled)

        # Apply initial state
        self._on_use_toggled(self._use_diam.isChecked(), value_off=0, value_on=30)

        # flow threshold
        self._flow_threshold = QPMSettingsSpinBox("Flow Threshold:", parent=self)
        self._flow_threshold.setDecimals(2)
        self._flow_threshold.setValue(0.4)
        self._flow_threshold.setToolTip(
            "Flow error threshold (all cells with errors below threshold are kept) (not used for 3D). Defaults to 0.4"
        )

        # cellprob threshold
        self._cellprob_threshold = QPMSettingsSpinBox(
            "Cellprob Threshold:", parent=self
        )
        self._cellprob_threshold.setDecimals(2)
        self._cellprob_threshold.setValue(0.0)
        self._cellprob_threshold.setToolTip(
            "All pixels with value above threshold kept for masks, decrease to find more and larger masks. Defaults to 0.0."
        )

        # min size
        self._min_size = QPMSettingsSpinBox("Min Size (pixels):", parent=self)
        self._min_size.setDecimals(0)
        self._min_size.setValue(15)
        self._min_size.setToolTip(
            "All ROIs below this size, in pixels, will be discarded. Defaults to 15."
        )

        # max size fraction
        self._max_size_fraction = QPMSettingsSpinBox("Max Size Fraction:", parent=self)
        self._max_size_fraction.setDecimals(2)
        self._max_size_fraction.setValue(0.4)
        self._max_size_fraction.setToolTip(
            "Masks larger than max_size_fraction of total image size are removed. Default is 0.4."
        )

        # label styling
        fixed_w = self._na_in._label.sizeHint().width()
        for widget in [
            self._diameter,
            self._flow_threshold,
            self._cellprob_threshold,
            self._min_size,
            self._max_size_fraction,
        ]:
            widget._label.setFixedWidth(fixed_w)
        self._input_dir._label.setFixedWidth(fixed_w)
        self._output_dir._label.setFixedWidth(fixed_w)
        r_lbl.setFixedWidth(fixed_w)
        i_lbl.setFixedWidth(fixed_w)

        # Analysis widget (QPM)
        analysis_wdg = QWidget(self)
        a_lbl = QLabel("Analysis to perform:", self)
        a_lbl.setFixedWidth(fixed_w)
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

        # analysis (phase)
        # analysis_phase_wdg = QWidget(self)
        # analysis_p_layout = QHBoxLayout(analysis_phase_wdg)
        # analysis_p_layout.setContentsMargins(0, 0, 0, 0)
        # analysis_p_layout.setSpacing(10)
        # analysis_p_layout.addStretch()

        # bottom widget
        # progress bar
        self._progress_bar = QProgressBar(self)
        self._progress_bar.setTextVisible(True)
        self._progress_bar.setStyleSheet(BAR_STYLESHEET)
        self._progress_bar.setFixedHeight(15)

        # buttons
        run_btn = QPushButton("Run")
        run_btn.setIcon(QIconifyIcon("streamline-sharp:startup-remix", color=GREEN))
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setIcon(
            QIconifyIcon("fluent-emoji-high-contrast:woman-gesturing-no", color=RED)
        )
        btns_layout = QHBoxLayout()
        btns_layout.setContentsMargins(0, 0, 0, 0)
        btns_layout.setSpacing(5)
        btns_layout.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        btns_layout.addWidget(run_btn)
        btns_layout.addWidget(cancel_btn)
        btns_layout.addWidget(self._progress_bar)
        run_btn.clicked.connect(self.run)
        cancel_btn.clicked.connect(self.cancel)

        # main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(5)
        main_layout.addWidget(create_divider_line("Input/Output Directories"))
        main_layout.addWidget(self._input_dir)
        main_layout.addWidget(self._output_dir)
        main_layout.addWidget(self._tabs)

        # qpm layout tab
        qpm_layout = QVBoxLayout(self.qpm_widget)
        qpm_layout.addWidget(create_divider_line("QPM Settings"))
        qpm_layout.addWidget(self._wav)
        qpm_layout.addWidget(self._mag)
        qpm_layout.addWidget(self._na)
        qpm_layout.addWidget(self._na_in)
        qpm_layout.addWidget(self._cam_pixel_size)
        # qpm_layout.addWidget(self._num_channels)
        qpm_layout.addWidget(rotation_wdg)
        qpm_layout.addWidget(invert_ph)
        qpm_layout.addWidget(create_divider_line("QPM Tikhonov Settings"))
        qpm_layout.addWidget(self._tikhonov_abs)
        qpm_layout.addWidget(self._tikhonov_ph)
        qpm_layout.addWidget(create_divider_line("Analysis"))
        qpm_layout.addWidget(analysis_wdg)

        # phc layout tab
        phc_layout = QVBoxLayout(self.phc_widget)
        phc_layout.addWidget(create_divider_line("Segmentation settings"))
        # phc_layout.addWidget(analysis_phase_wdg)

        hbox_diam = QHBoxLayout()
        hbox_diam.addWidget(self._use_diam)
        hbox_diam.addWidget(self._diameter)

        # phc_layout.addWidget(h_box_widget)
        phc_layout.addLayout(hbox_diam)

        phc_layout.addWidget(self._flow_threshold)
        phc_layout.addWidget(self._cellprob_threshold)
        phc_layout.addWidget(self._min_size)
        phc_layout.addWidget(self._max_size_fraction)
        phc_layout.addStretch()

        # add an image widget after the stretch
        img_label = QLabel(self)

        arr = tifffile.imread(CAT_PATH / "cat.tif")

        h, w = arr.shape
        qimg = QImage(arr.data, w, h, w, QImage.Format.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimg)
        img_label.setPixmap(pixmap)
        img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        phc_layout.addWidget(img_label)

        main_layout.addWidget(create_divider_line())
        main_layout.addLayout(btns_layout)

        # TO REMOVE, JUST FOR TESTING
        input = TEST_DATA / "input"
        output = TEST_DATA / "output"
        self._input_dir.setValue(input)
        self._output_dir.setValue(output)
        # END TO REMOVE

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

        if self._tabs.currentIndex() == 0:  # QPM tab
            self._worker = create_worker(
                self._run_qpm,
                _start_thread=True,
                _connect={
                    "yielded": self._update_progress,
                    "finished": self._on_processing_finished,
                    "errored": self._on_error,
                },
            )

        if self._tabs.currentIndex() == 1:  # phase contrast tab
            self._worker = create_worker(
                self._run_phase_contrast,
                _start_thread=True,
                _connect={
                    "yielded": self._update_progress,
                    "finished": self._on_processing_finished,
                    "errored": self._on_error,
                },
            )

    def _on_use_toggled(self, checked: bool, value_off=0, value_on=30) -> None:
        """Enable or disable the diameter spin box based on the checkbox state."""
        self._diameter.setEnabled(checked)
        if checked:
            self._diameter.setValue(value_on)
        elif not checked:
            self._diameter.setValue(value_off)
            pass

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

    def _on_processing_finished(self) -> None:
        """Called when processing is finished."""
        self._cancel_requested = False
        print("Processing completed!")

    def _on_error(self) -> None:
        """Called when an error occurs during processing."""
        self._cancel_requested = False
        print("Processing failed!")
        # clear the progress bar
        self._progress_bar.setValue(0)
        self._progress_bar.setFormat("")

    def _run_qpm(self) -> Generator[dict, None, None]:
        """Run the QPM processing."""
        if not self._input_dir.value():
            return

        if not self._output_dir.value():
            show_error_dialog(self, "Output directory is not set.")
            return

        rotations = self._parse_rotation()

        num_files = self._get_total_number_of_files()

        # Initialize progress bar
        yield {"type": "init", "total": num_files}

        # understand if the solver should be initialized at each image or not
        # self._dpc_solver = None

        current_file = 0
        path = Path(self._input_dir.value())
        for item in path.iterdir():
            if self._cancel_requested:
                return

            if item.is_file() and item.suffix in {".tif", ".tiff"}:
                name = item.stem.replace(".ome", "")

                image = tifffile.imread(item)

                if not self._validate_image(image):
                    continue

                out = Path(self._output_dir.value()) / f"{name}{PROCESSED}"
                out.mkdir(parents=True, exist_ok=True)

                ph, seg = None, None
                if self._qpm_reconstruct_cbox.isChecked():
                    self._dpc_solver = None
                    ph = self._reconstruct_qpm(image, name, rotations, out)
                if self._segment_cbox.isChecked():
                    seg = self._segment_file(ph, name, out)
                if ph is not None and seg is not None:
                    self._generate_csv_file(seg, ph, name, out)

                current_file += 1
                yield {"type": "update", "current": current_file}

            elif item.is_dir():
                for tif_file in item.glob("*.tif"):
                    name = tif_file.stem.replace(".ome", "")

                    image = tifffile.imread(tif_file)

                    if not self._validate_image(image):
                        continue

                    out = Path(self._output_dir.value()) / f"{name}{PROCESSED}"
                    out.mkdir(parents=True, exist_ok=True)

                    ph, seg = None, None
                    if self._qpm_reconstruct_cbox.isChecked():
                        self._dpc_solver = None
                        ph = self._reconstruct_qpm(image, tif_file.stem, rotations, out)
                    if self._segment_cbox.isChecked():
                        seg = self._segment_file(ph, tif_file.stem, out)
                    if ph is not None and seg is not None:
                        self._generate_csv_file(seg, ph, tif_file.stem, out)

                    current_file += 1
                    yield {"type": "update", "current": current_file}

    def _run_phase_contrast(self) -> Generator[dict, None, None]:
        """Run the phase contrast processing."""
        if not self._input_dir.value():
            return

        if not self._output_dir.value():
            show_error_dialog(self, "Output directory is not set.")
            return

        num_files = self._get_total_number_of_files()

        # Initialize progress bar
        yield {"type": "init", "total": num_files}

        current_file = 0
        path = Path(self._input_dir.value())
        for item in path.iterdir():
            if self._cancel_requested:
                return

            if item.is_file() and item.suffix in {".tif", ".tiff"}:
                name = item.stem.replace(".ome", "")

                image = tifffile.imread(item)

                if not self._validate_image(image):
                    continue

                out = Path(self._output_dir.value()) / f"{name}{PROCESSED}"
                out.mkdir(parents=True, exist_ok=True)

                seg = self._segment_file(image, name, out)
                if seg is not None:
                    self._generate_csv_file(seg, image, name, out)

                current_file += 1
                yield {"type": "update", "current": current_file}

            elif item.is_dir():
                for tif_file in item.glob("*.tif"):
                    name = tif_file.stem.replace(".ome", "")

                    image = tifffile.imread(tif_file)

                    if not self._validate_image(image):
                        continue

                    out = Path(self._output_dir.value()) / f"{name}{PROCESSED}"
                    out.mkdir(parents=True, exist_ok=True)

                    seg = self._segment_file(image, tif_file.stem, out)
                    if seg is not None:
                        self._generate_csv_file(seg, image, tif_file.stem, out)

                    current_file += 1
                    yield {"type": "update", "current": current_file}

    def _validate_image(self, image: np.ndarray) -> bool:
        """Validate the input image."""
        if image.ndim != 3:
            show_error_dialog(self, "The file should have 3 dimensions (C, H, W).")
            return False
        if image.shape[0] != 4:
            show_error_dialog(
                self, "The file should have 4 channels, one per illumination angle."
            )
            return False
        return True

    def _parse_rotation(self) -> list[float]:
        """Parse the rotation angles from the input."""
        try:
            return [float(angle.strip()) for angle in self._rotation.text().split(",")]
        except ValueError:
            show_error_dialog(self, "Invalid rotation angles format.")
            return []

    def _get_total_number_of_files(self) -> int:
        """Get the total number of files to process."""
        path = Path(self._input_dir.value())
        total_files = 0
        for item in path.iterdir():
            if item.is_file() and item.suffix in {".tif", ".tiff"}:
                total_files += 1
            elif item.is_dir():
                total_files += len(list(item.glob("*.tif")))
        return total_files

    def _segment_file(
        self, image: np.ndarray | None, name: str, output_dir: Path
    ) -> np.ndarray | None:
        """Process a single TIF file."""
        if image is None:
            return None

        print(f"\nProcessing {name}...")

        if self._cancel_requested:
            return None

        # run segmentation
        print("Running CellposeSAM segmentation...")
        diameter = self._diameter.value() if self._use_diam.isChecked() else None
        flow_threshold = self._flow_threshold.value()
        cellprob_threshold = self._cellprob_threshold.value()
        min_size = int(self._min_size.value())
        max_size_fraction = self._max_size_fraction.value()

        labels, _, _ = self._cp2.eval(
            image,
            diameter=diameter,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            min_size=min_size,
            max_size_fraction=max_size_fraction,
        )

        # save the labels
        print("Saving labels...")
        io.imsave(output_dir / f"{name}_labels.tif", labels)
        # This I would remove later on or make it optional.
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        ax[0].imshow(image, cmap="gray")
        ax[0].set_title("Phase Image")
        ax[0].axis("off")
        ax[1].imshow(labels, cmap="nipy_spectral")
        ax[1].set_title("Labels")
        ax[1].axis("off")
        plt.tight_layout()
        plt.savefig(output_dir / f"{name}_labels.png", dpi=150)
        plt.close()
        print(f"Saved labels for {name}")
        return labels

    def _reconstruct_qpm(
        self, image: np.ndarray, name: str, rotations: list[float], output_dir: Path
    ) -> np.ndarray | None:
        """Reconstruct the QPM image.

        Code form Laura Waller Lab: https://github.com/Waller-Lab/DPC/tree/master/python_code
        """
        print(f"Reconstructing QPM for {name}...")

        if self._dpc_solver is None:
            print("Initializing DPCSolver...")
            self._dpc_solver = DPCSolver(
                image,
                self._wav.value(),
                self._na.value(),
                self._na_in.value(),
                self._cam_pixel_size.value() / self._mag.value(),
                rotations,
                # dpc_num=int(self._num_channels.value()),
                dpc_num=4,
            )
        else:
            self._dpc_solver.set_dpc_imgs(image)
            self._dpc_solver.normalization()

        if self._cancel_requested:
            return None

        # solve DPC Deconvoltion Problems
        # parameters for Tikhonov regurlarization [u:absorption, p:phase]
        # ((need to tune this based on SNR)
        print("Solving DPC Deconvoltion Problems...")
        self._dpc_solver.setRegularizationParameters(
            reg_u=self._tikhonov_abs.value(), reg_p=self._tikhonov_ph.value()
        )
        dpc_result = self._dpc_solver.solve(method="Tikhonov")

        print("Saving QPM results...")
        abs = dpc_result[0].real.astype("float32")
        ph = dpc_result[0].imag.astype("float32")
        if self._invert_ph.isChecked():
            ph = -ph
        tifffile.imwrite(output_dir / f"frame_{name}_abs.tif", abs, imagej=True)
        tifffile.imwrite(output_dir / f"frame_{name}_ph.tif", ph, imagej=True)
        print(f"Saved QPM results for {name}")
        return ph

    def _generate_csv_file(
        self, labels: np.ndarray, phase_image: np.ndarray, name: str, output_dir: Path
    ) -> None:
        """Generate a CSV file with measurements."""
        name = name.replace(".ome", "")
        print(f"Generating CSV file for {name}...")

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
        print(f"Saving CSV file for {name}...")
        props_df.to_csv(output_dir / f"{name}_measurements.csv", index=False)
        print(f"Saved CSV file for {name}")

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
