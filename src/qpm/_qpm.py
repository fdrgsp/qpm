from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
from PyQt6.QtWidgets import (
    QWidget,
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
    QPMSettingsSpinBox,
    create_divider_line,
    show_error_dialog,
)
from ._dpc_algorithm import DPCSolver
from superqt import QIconifyIcon
from ._segmentation import CellposeSAMSegmentation
import tifffile
from cellpose import io
from superqt.utils import create_worker, GeneratorWorker, FunctionWorker

RED = "#C33"
GREEN = "#00FF00"
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


class QPMWidget(QWidget):
    """The QPM widget."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)

        self.setWindowTitle("QPM Widget")
        self.resize(600, 400)

        self._worker: FunctionWorker | GeneratorWorker | None = None

        self._dpc_solver: DPCSolver | None = None

        # segmentation
        self._cp = CellposeSAMSegmentation()

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
        self._num_channels = QPMSettingsSpinBox("Number of Channels:", parent=self)
        self._num_channels.setDecimals(0)
        self._num_channels.setValue(4)

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
            self._num_channels,
            self._tikhonov_abs,
            self._tikhonov_ph,
        ]:
            widget._label.setFixedWidth(fixed_w)
        self._input_dir._label.setFixedWidth(fixed_w)
        self._output_dir._label.setFixedWidth(fixed_w)
        r_lbl.setFixedWidth(fixed_w)
        i_lbl.setFixedWidth(fixed_w)

        # progress bar
        self._progress_bar = QProgressBar(self)
        self._progress_bar.setTextVisible(True)
        # self._progress_bar.setVisible(False)
        self._progress_bar.setStyleSheet(BAR_STYLESHEET)
        self._progress_bar.setFixedHeight(15)

        # bottom widget
        run_btn = QPushButton("Run")
        run_btn.setIcon(QIconifyIcon("mdi:play", color=GREEN))
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setIcon(QIconifyIcon("mdi:stop", color=RED))
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
        main_layout.addWidget(create_divider_line("QPM Settings"))
        main_layout.addWidget(self._wav)
        main_layout.addWidget(self._mag)
        main_layout.addWidget(self._na)
        main_layout.addWidget(self._na_in)
        main_layout.addWidget(self._cam_pixel_size)
        main_layout.addWidget(self._num_channels)
        main_layout.addWidget(rotation_wdg)
        main_layout.addWidget(invert_ph)
        main_layout.addWidget(create_divider_line("Tikhonov Settings"))
        main_layout.addWidget(self._tikhonov_abs)
        main_layout.addWidget(self._tikhonov_ph)
        main_layout.addWidget(create_divider_line())
        main_layout.addLayout(btns_layout)

        # TO REMOVE, JUST FOR TESTING
        self._input_dir.setValue("/Users/fdrgsp/Desktop/qpm/test")
        self._output_dir.setValue("/Users/fdrgsp/Desktop/t")
        # END TO REMOVE

    def cancel(self) -> None:
        """Cancel the QPM processing."""
        if self._worker is None or not self._worker.is_running:
            return
        self._worker.quit()

    def run(self) -> None:
        """Run the QPM processing in a separate thread."""
        self._worker = create_worker(
            self._run,
            _start_thread=True,
            _connect={
                "yielded": self._update_progress,
                "finished": self._on_processing_finished,
            },
        )

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
        # self._progress_bar.setVisible(False)
        print("Processing completed!")

    def _run(self) -> Generator[dict, None, None]:
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

            if item.is_file() and item.suffix in {".tif", ".tiff"}:
                self._dpc_solver = None
                image = tifffile.imread(item)

                # create folder for the results
                name = item.stem.replace(".ome", "")
                output_dir = Path(self._output_dir.value()) / f"{name}{PROCESSED}"
                output_dir.mkdir(parents=True, exist_ok=True)

                # seg = self._segment_file(image, name)
                ph = self._reconstruct_qpm(image, name, rotations)
                # self._generate_csv_file(seg, ph, name)

                current_file += 1
                yield {"type": "update", "current": current_file}

            elif item.is_dir():
                for tif_file in item.glob("*.tif"):
                    self._dpc_solver = None
                    image = tifffile.imread(tif_file)
                    # seg = self._segment_file(image, tif_file.stem)
                    ph = self._reconstruct_qpm(image, tif_file.stem, rotations)
                    # self._generate_csv_file(seg, ph, tif_file.stem)

                    current_file += 1
                    yield {"type": "update", "current": current_file}

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

    def _segment_file(self, image: np.ndarray, name: str) -> np.ndarray:
        """Process a single TIF file."""
        print(f"\nProcessing {name}...")
        assert image.ndim == 3, "The file should have 3 dimensions (C, H, W)."
        assert (
            image.shape[0] == 4
        ), "The file should have 4 channels, one per illumination angle."

        # run segmentation
        print("Running CellposeSAM segmentation...")
        max_image = np.max(image, axis=0)
        labels, _ = self._cp.eval(max_image)

        # save the labels
        print("Saving labels...")
        output_dir = Path(self._output_dir.value()) / f"{name}{PROCESSED}"
        io.imsave(output_dir / f"{name}_labels.tif", labels)
        io.imsave(output_dir / f"{name}_max.tif", max_image)
        # This I would remove later on or make it optional.
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        ax[0].imshow(max_image, cmap="gray")
        ax[0].set_title("Raw Max Projection")
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
        self, image: np.ndarray, name: str, rotations: list[float]
    ) -> np.ndarray:
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
                dpc_num=int(self._num_channels.value()),
            )
        else:
            self._dpc_solver.set_dpc_imgs(image)
            self._dpc_solver.normalization()

        # solve DPC Deconvoltion Problems
        # parameters for Tikhonov regurlarization [u:absorption, p:phase]
        # ((need to tune this based on SNR)
        print("Solving DPC Deconvoltion Problems...")
        self._dpc_solver.setRegularizationParameters(
            reg_u=self._tikhonov_abs.value(), reg_p=self._tikhonov_ph.value()
        )
        dpc_result = self._dpc_solver.solve(method="Tikhonov")

        print("Saving QPM results...")
        output_dir = Path(self._output_dir.value()) / f"{name}{PROCESSED}"
        abs = dpc_result[0].real.astype("float32")
        ph = dpc_result[0].imag.astype("float32")
        if self._invert_ph.isChecked():
            ph = -ph
        tifffile.imwrite(output_dir / f"frame_{name}_abs.tif", abs, imagej=True)
        tifffile.imwrite(output_dir / f"frame_{name}_ph.tif", ph, imagej=True)
        print(f"Saved QPM results for {name}")
        return ph

    def _generate_csv_file(
        self, labels: np.ndarray, phase_image: np.ndarray, name: str
    ) -> None:
        """Generate a CSV file with measurements."""
        name = name.replace(".ome", "")
        print(f"Generating CSV file for {name}...")

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
        props_df = pd.DataFrame(props_dict)
        print(f"Saving CSV file for {name}...")
        output_dir = Path(self._output_dir.value()) / f"{name}{PROCESSED}"
        props_df.to_csv(output_dir / f"{name}_measurements.csv", index=False)
        print(f"Saved CSV file for {name}")

        # Temporary plots for testing
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
