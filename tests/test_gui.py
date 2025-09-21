import sys
import numpy as np
import tifffile
from pathlib import Path
import pytest

from PyQt6.QtWidgets import QApplication

import qpm._qpm as qpm_mod
from qpm._qpm import QPMWidget, QPM_PROCESSED, PHC_PROCESSED

# Create a QApplication for widgets
_app = QApplication.instance() or QApplication(sys.argv)


class DummySegmentation:
    def __init__(self, *a, **k):
        self.params = {}

    def set_parameters(self, **kwargs):
        self.params.update(kwargs)

    def eval(self, image):
        # return a label image with same H,W (support either 2D or 3D input)
        if image is None:
            return None, None
        if image.ndim == 3:
            h, w = image.shape[1], image.shape[2]
        else:
            h, w = image.shape[0], image.shape[1]
        labels = np.zeros((h, w), dtype=np.int16)
        return labels, None


@pytest.fixture(autouse=True)
def patch_segmentation(monkeypatch, tmp_path):
    """Monkeypatch heavy components before each test."""
    # Replace CellposeSAMSegmentation with dummy to avoid heavy model init
    monkeypatch.setattr(qpm_mod, "CellposeSAMSegmentation", DummySegmentation)
    # Prevent GUI error dialogs from blocking tests
    monkeypatch.setattr(qpm_mod, "show_error_dialog", lambda *a, **k: None)
    yield


def create_widget(tmp_path: Path) -> QPMWidget:
    """Helper: create a QPMWidget with test input/output set."""
    w = QPMWidget()
    # point input/output to tmp paths
    w._input_dir.setValue(tmp_path)
    w._output_dir.setValue(tmp_path)
    return w


def test_validate_qpm_and_phc():
    w = create_widget(Path("."))

    # QPM validation: correct shape
    img_qpm = np.zeros((4, 10, 12), dtype=np.float32)
    ok, msg = w._validate_qpm_image(img_qpm)
    assert ok is True
    assert msg == ""

    # QPM invalid: wrong ndim
    img_wrong = np.zeros((10, 12), dtype=np.float32)
    ok, msg = w._validate_qpm_image(img_wrong)
    assert ok is False
    assert "3 dimensions" in msg

    # QPM invalid: wrong channel count
    img_bad_ch = np.zeros((3, 10, 12), dtype=np.float32)
    ok, msg = w._validate_qpm_image(img_bad_ch)
    assert ok is False
    assert "4 channels" in msg

    # PHC validation: correct
    img_phc = np.zeros((20, 30), dtype=np.float32)
    ok, msg = w._validate_phc_image(img_phc)
    assert ok is True

    # PHC invalid: 3D
    ok, msg = w._validate_phc_image(img_qpm)
    assert ok is False
    assert "2 dimensions" in msg


def test_get_sorted_files_and_dirs(tmp_path: Path):
    # create files and subfolder
    f1 = tmp_path / "a.tif"
    f2 = tmp_path / "b.tif"
    tifffile.imwrite(f1, np.zeros((10, 10), dtype=np.uint8), photometric="minisblack")
    tifffile.imwrite(f2, np.zeros((10, 10), dtype=np.uint8), photometric="minisblack")
    sub = tmp_path / "sub"
    sub.mkdir()
    f3 = sub / "c.tif"
    tifffile.imwrite(f3, np.zeros((10, 10), dtype=np.uint8), photometric="minisblack")

    w = create_widget(tmp_path)
    sorted_items, total_files = w._get_sorted_files_and_dirs()
    assert total_files == 3
    assert len(sorted_items) == 3  # 2 files + 1 directory
    # Check that items are sorted by name
    assert sorted_items[0].name == "a.tif"
    assert sorted_items[1].name == "b.tif"
    assert sorted_items[2].name == "sub"


def test_process_single_qpm_image_success_and_failure(tmp_path: Path):
    w = create_widget(tmp_path)
    # disable heavy operations
    w._qpm_reconstruct_cbox.setChecked(False)
    w._segment_cbox.setChecked(False)

    # valid qpm image (4, H, W)
    valid_data = np.zeros((4, 8, 9), dtype=np.float32)
    ok, msg = w._process_single_qpm_image(valid_data, "valid_qpm", [0, 90, 180, 270])
    assert ok is True
    assert msg == ""
    outdir = tmp_path / f"valid_qpm{QPM_PROCESSED}"
    assert outdir.exists() and outdir.is_dir()

    # invalid qpm image (wrong channels)
    invalid_data = np.zeros((2, 5, 5), dtype=np.float32)
    ok, msg = w._process_single_qpm_image(
        invalid_data, "invalid_qpm", [0, 90, 180, 270]
    )
    assert ok is False
    assert "invalid_qpm" in msg
    assert "4 channels" in msg


def test_process_single_phc_image_success_and_failure(tmp_path: Path):
    w = create_widget(tmp_path)

    # valid phc image (2D)
    valid_data = np.zeros((15, 16), dtype=np.uint8)
    ok, msg = w._process_single_phc_image(valid_data, "valid_phc")
    assert ok is True
    assert msg == ""
    outdir = tmp_path / f"valid_phc{PHC_PROCESSED}"
    assert outdir.exists() and outdir.is_dir()

    # invalid (3D)
    invalid_data = np.zeros((3, 5, 5), dtype=np.uint8)
    ok, msg = w._process_single_phc_image(invalid_data, "bad_phc")
    assert ok is False
    assert "bad_phc" in msg
    assert "2 dimensions" in msg


def test_parse_rotation_valid_and_invalid(monkeypatch):
    w = create_widget(Path("."))
    w._rotation.setText("180, 90, 270, 0")
    rotations = w._parse_rotation()
    assert isinstance(rotations, list)
    assert len(rotations) == 4

    # invalid rotation string -> returns [] (show_error_dialog is patched to no-op)
    w._rotation.setText("bad,data")
    rotations = w._parse_rotation()
    assert rotations == []


def test_segment_file_none_and_return(tmp_path: Path):
    w = create_widget(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    assert w._segment_file(None, "none", out) is None

    # when provided an image, DummySegmentation will be used and files will be written
    img = np.zeros((20, 20), dtype=np.uint8)
    labels = w._segment_file(img, "img", out)
    assert labels is not None
    assert (out / "img_labels.tif").exists()
    assert (out / "img_labels.png").exists()


def test_qpm_tif_generator_simple_tiff(tmp_path: Path):
    """Test QPM generator with a simple TIFF file (no OME metadata)."""
    # Create a simple TIFF file
    simple_tif = tmp_path / "simple.tif"
    data = np.zeros((4, 8, 9), dtype=np.float32)
    tifffile.imwrite(simple_tif, data, photometric="minisblack")

    w = create_widget(tmp_path)
    w._qpm_reconstruct_cbox.setChecked(False)
    w._segment_cbox.setChecked(False)

    # Test the generator
    updates = list(w._process_qpm_tif_generator(simple_tif, [0, 90, 180, 270]))
    assert len(updates) == 1
    assert updates[0]["type"] == "progress"


def test_phc_tif_generator_simple_tiff(tmp_path: Path):
    """Test PHC generator with a simple TIFF file (no OME metadata)."""
    # Create a simple TIFF file
    simple_tif = tmp_path / "simple.tif"
    data = np.zeros((15, 16), dtype=np.uint8)
    tifffile.imwrite(simple_tif, data, photometric="minisblack")

    w = create_widget(tmp_path)

    # Test the generator
    updates = list(w._process_phc_tif_generator(simple_tif))
    assert len(updates) == 1
    assert updates[0]["type"] == "progress"


def test_qpm_tif_generator_validation_error(tmp_path: Path):
    """Test QPM generator with invalid data."""
    # Create an invalid TIFF file (wrong channels)
    invalid_tif = tmp_path / "invalid.tif"
    data = np.zeros((2, 8, 9), dtype=np.float32)  # Only 2 channels instead of 4
    tifffile.imwrite(invalid_tif, data, photometric="minisblack")

    w = create_widget(tmp_path)

    # Test the generator
    updates = list(w._process_qpm_tif_generator(invalid_tif, [0, 90, 180, 270]))
    assert len(updates) == 1
    assert updates[0]["type"] == "error"
    assert "invalid" in updates[0]["message"]
    assert "4 channels" in updates[0]["message"]


def test_phc_tif_generator_validation_error(tmp_path: Path):
    """Test PHC generator with invalid data."""
    # Create an invalid TIFF file (3D instead of 2D)
    invalid_tif = tmp_path / "invalid.tif"
    data = np.zeros((3, 8, 9), dtype=np.uint8)  # 3D instead of 2D
    tifffile.imwrite(invalid_tif, data, photometric="minisblack")

    w = create_widget(tmp_path)

    # Test the generator
    updates = list(w._process_phc_tif_generator(invalid_tif))
    assert len(updates) == 1
    assert updates[0]["type"] == "error"
    assert "invalid" in updates[0]["message"]
    assert "2 dimensions" in updates[0]["message"]


def test_qpm_tif_generator_multi_position_mock(tmp_path: Path, monkeypatch):
    """Test QPM generator with multi-position OME-TIFF handling (mocked)."""

    class MockPosition:
        def __init__(self, name, data):
            self.name = name
            self._data = data

        def asarray(self):
            return self._data

    class MockTiffFile:
        def __init__(self, positions):
            self.series = positions

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    # Mock tifffile.TiffFile to simulate multi-position OME-TIFF
    def mock_tiff_file(path):
        # Create mock positions with valid QPM data
        pos1_data = np.zeros((4, 8, 9), dtype=np.float32)
        pos2_data = np.zeros((4, 8, 9), dtype=np.float32)
        return MockTiffFile(
            [
                MockPosition("Position_0", pos1_data),
                MockPosition("Position_1", pos2_data),
            ]
        )

    monkeypatch.setattr("tifffile.TiffFile", mock_tiff_file)

    w = create_widget(tmp_path)
    w._qpm_reconstruct_cbox.setChecked(False)
    w._segment_cbox.setChecked(False)

    # Create a dummy file (content doesn't matter since we're mocking TiffFile)
    multi_pos_tif = tmp_path / "multi_pos.tif"
    multi_pos_tif.write_bytes(b"dummy")

    # Test the generator - should yield progress for each position
    updates = list(w._process_qpm_tif_generator(multi_pos_tif, [0, 90, 180, 270]))
    assert len(updates) == 2  # One progress update per position
    assert all(update["type"] == "progress" for update in updates)

    # Check that position names were added to skip files
    assert "Position_0" in w._skip_files
    assert "Position_1" in w._skip_files


def test_phc_tif_generator_multi_position_mock(tmp_path: Path, monkeypatch):
    """Test PHC generator with multi-position OME-TIFF handling (mocked)."""

    class MockPosition:
        def __init__(self, name, data):
            self.name = name
            self._data = data

        def asarray(self):
            return self._data

    class MockTiffFile:
        def __init__(self, positions):
            self.series = positions

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    # Mock tifffile.TiffFile to simulate multi-position OME-TIFF
    def mock_tiff_file(path):
        # Create mock positions with valid PHC data
        pos1_data = np.zeros((15, 16), dtype=np.uint8)
        pos2_data = np.zeros((15, 16), dtype=np.uint8)
        return MockTiffFile(
            [
                MockPosition("Position_0", pos1_data),
                MockPosition("Position_1", pos2_data),
            ]
        )

    monkeypatch.setattr("tifffile.TiffFile", mock_tiff_file)

    w = create_widget(tmp_path)

    # Create a dummy file (content doesn't matter since we're mocking TiffFile)
    multi_pos_tif = tmp_path / "multi_pos.tif"
    multi_pos_tif.write_bytes(b"dummy")

    # Test the generator - should yield progress for each position
    updates = list(w._process_phc_tif_generator(multi_pos_tif))
    assert len(updates) == 2  # One progress update per position
    assert all(update["type"] == "progress" for update in updates)

    # Check that position names were added to skip files
    assert "Position_0" in w._skip_files
    assert "Position_1" in w._skip_files


def test_widget_initialization():
    """Test widget initialization and basic properties."""
    w = QPMWidget()

    # Check basic properties
    assert w.windowTitle() == "QPM Widget"
    assert w._cancel_requested is False
    assert w._worker is None
    assert w._dpc_solver is None
    assert isinstance(w._skip_files, list)
    assert len(w._skip_files) == 0

    # Check UI elements exist
    assert w._tabwidget.count() == 2
    assert w._tabwidget.tabText(0) == "QPM"
    assert w._tabwidget.tabText(1) == "Phase Contrast Segmentation"

    # Check default values
    assert w._wav.value() == 0.530
    assert w._mag.value() == 40
    assert w._na.value() == 0.75
    assert w._na_in.value() == 0.0
    assert w._cam_pixel_size.value() == 6.5
    assert w._rotation.text() == "180, 90, 270, 0"
    assert w._invert_ph.isChecked() is False
    assert w._tikhonov_abs.value() == 0.1
    assert w._tikhonov_ph.value() == 0.005
    assert w._segment_cbox.isChecked() is True
    assert w._qpm_reconstruct_cbox.isChecked() is True


def test_enable_disable_widgets(tmp_path: Path):
    """Test enabling/disabling widgets."""
    w = create_widget(tmp_path)

    # Test disable
    w._enable(False)
    assert not w._input_dir.isEnabled()
    assert not w._output_dir.isEnabled()
    assert not w._seg_group.isEnabled()
    assert not w._tabwidget.isEnabled()

    # Test enable
    w._enable(True)
    assert w._input_dir.isEnabled()
    assert w._output_dir.isEnabled()
    assert w._seg_group.isEnabled()
    assert w._tabwidget.isEnabled()


def test_rename_run_buttons(tmp_path: Path):
    """Test run button renaming based on tab selection."""
    w = create_widget(tmp_path)

    # QPM tab (index 0)
    w._tabwidget.setCurrentIndex(0)
    w._rename_run_buttons()
    assert w._run_btn.text() == "Run QPM Processing"

    # Phase contrast tab (index 1)
    w._tabwidget.setCurrentIndex(1)
    w._rename_run_buttons()
    assert w._run_btn.text() == "Run PhC Segmentation"


def test_update_progress(tmp_path: Path):
    """Test progress bar updates."""
    w = create_widget(tmp_path)

    # Test init
    w._update_progress({"type": "init", "total": 10})
    assert w._progress_bar.maximum() == 10
    assert w._progress_bar.value() == 0
    assert w._progress_bar.format() == "0/10"
    # Note: isVisible() may not work in headless testing, so we skip this check

    # Test update
    w._update_progress({"type": "update", "current": 5})
    assert w._progress_bar.value() == 5
    assert w._progress_bar.format() == "5/10"

    # Test error (won't show dialog due to patching)
    w._update_progress({"type": "error", "message": "Test error"})

    # Test validation errors (won't show dialog due to patching)
    w._update_progress(
        {"type": "validation_errors", "message": "Test validation error"}
    )


def test_cancel_processing(tmp_path: Path):
    """Test canceling processing."""
    w = create_widget(tmp_path)

    # Test cancel when no worker
    w.cancel()  # Should not raise any error
    assert w._cancel_requested is False

    # Test with mock worker
    class MockWorker:
        def __init__(self, is_running=True):
            self.is_running = is_running
            self.quit_called = False

        def quit(self):
            self.quit_called = True

    # Test cancel with running worker
    w._worker = MockWorker(is_running=True)
    w.cancel()
    assert w._cancel_requested is True
    assert w._worker.quit_called is True
    assert w._progress_bar.value() == 0
    assert w._progress_bar.format() == ""

    # Test cancel with non-running worker
    w._cancel_requested = False
    w._worker = MockWorker(is_running=False)
    w.cancel()
    assert w._cancel_requested is False  # Should not change


def test_on_processing_finished(tmp_path: Path):
    """Test processing finished callback."""
    w = create_widget(tmp_path)
    w._cancel_requested = True
    w._enable(False)

    w._on_processing_finished()
    assert w._cancel_requested is False
    # Note: _enable(True) would be called but we can't easily test it without mocking


def test_on_error(tmp_path: Path):
    """Test error handling callback."""
    w = create_widget(tmp_path)
    w._cancel_requested = True
    w._progress_bar.setValue(50)
    w._progress_bar.setFormat("50/100")
    w._enable(False)

    # Test with a dummy exception
    try:
        raise ValueError("Test error")
    except ValueError as e:
        w._on_error(e)

    assert w._cancel_requested is False
    assert w._progress_bar.value() == 0
    assert w._progress_bar.format() == ""


def test_generate_csv_file(tmp_path: Path):
    """Test CSV file generation."""
    w = create_widget(tmp_path)

    # Create mock labels and phase image
    labels = np.array(
        [[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 0, 0], [2, 2, 0, 0]], dtype=np.int32
    )

    phase_image = np.array(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.2, 0.3, 0.4, 0.5],
            [0.5, 0.6, 0.1, 0.2],
            [0.6, 0.7, 0.2, 0.3],
        ],
        dtype=np.float32,
    )

    output_dir = tmp_path / "csv_test"
    output_dir.mkdir()

    # Test CSV generation
    w._generate_csv_file(labels, phase_image, "test_image", output_dir)

    # Check that files were created
    assert (output_dir / "test_image_measurements.csv").exists()
    assert (output_dir / "test_image_dry_mass_distribution.png").exists()
    assert (output_dir / "test_image_area_distribution.png").exists()
    assert (output_dir / "test_image_eccentricity_distribution.png").exists()
    assert (output_dir / "test_image_axis_major_length_distribution.png").exists()

    # Check CSV content
    import pandas as pd

    df = pd.read_csv(output_dir / "test_image_measurements.csv")
    assert "dry_mass" in df.columns
    assert "intensity_sum" in df.columns
    assert len(df) == 2  # Two regions (labels 1 and 2)


def test_generate_csv_file_with_cancel(tmp_path: Path):
    """Test CSV generation with cancel requested."""
    w = create_widget(tmp_path)
    w._cancel_requested = True

    labels = np.array([[1, 1], [1, 1]], dtype=np.int32)
    phase_image = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)

    output_dir = tmp_path / "csv_cancel_test"
    output_dir.mkdir()

    result = w._generate_csv_file(labels, phase_image, "test", output_dir)
    assert result is None  # Should return early due to cancel


def test_extract_and_add_dry_mass_calculation(tmp_path: Path):
    """Test dry mass calculation."""
    w = create_widget(tmp_path)

    # Simple test data
    labels = np.array([[1, 1], [2, 2]], dtype=np.int32)
    phase_image = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)

    props_dict = {"label": [1, 2]}

    w._extract_and_add_dry_mass_calculation(labels, phase_image, props_dict)

    assert "intensity_sum" in props_dict
    assert "dry_mass" in props_dict
    assert len(props_dict["intensity_sum"]) == 2
    assert len(props_dict["dry_mass"]) == 2
    assert all(dm > 0 for dm in props_dict["dry_mass"])  # Should be positive values


def test_run_qpm_missing_directories(tmp_path: Path):
    """Test QPM processing with missing input/output directories."""
    w = create_widget(tmp_path)

    # Test missing input directory
    w._input_dir.setValue("")
    updates = list(w._run_qpm())
    assert len(updates) == 0  # Should return early

    # Test missing output directory
    w._input_dir.setValue(tmp_path)
    w._output_dir.setValue("")
    updates = list(w._run_qpm())
    assert len(updates) == 1
    assert updates[0]["type"] == "error"
    assert "Output directory is not set" in updates[0]["message"]


def test_run_qpm_invalid_rotations(tmp_path: Path, monkeypatch):
    """Test QPM processing with invalid rotation format."""
    w = create_widget(tmp_path)

    # Mock show_error_dialog to avoid GUI dialogs
    error_called = []

    def mock_error_dialog(*args, **kwargs):
        error_called.append(True)

    monkeypatch.setattr(qpm_mod, "show_error_dialog", mock_error_dialog)

    w._rotation.setText("invalid,rotation,format")
    updates = list(w._run_qpm())
    assert len(updates) == 1
    assert updates[0]["type"] == "error"
    assert "Invalid rotation angles format" in updates[0]["message"]


def test_run_phc_missing_directories(tmp_path: Path):
    """Test PHC processing with missing directories."""
    w = create_widget(tmp_path)

    # Test missing input directory
    w._input_dir.setValue("")
    updates = list(w._run_phc())
    assert len(updates) == 0  # Should return early

    # Test missing output directory
    w._input_dir.setValue(tmp_path)
    w._output_dir.setValue("")
    updates = list(w._run_phc())
    assert len(updates) == 1
    assert updates[0]["type"] == "error"
    assert "Output directory is not set" in updates[0]["message"]


def test_skip_files_functionality(tmp_path: Path):
    """Test skip files functionality."""
    w = create_widget(tmp_path)

    # Create test files first
    f1 = tmp_path / "file1.tif"
    f2 = tmp_path / "file2.tif"  # This should be skipped
    f3 = tmp_path / "file3.tif"

    for f in [f1, f2, f3]:
        data = np.zeros((15, 16), dtype=np.uint8)
        tifffile.imwrite(f, data, photometric="minisblack")

    # Add files to skip AFTER they exist (stem without .ome extension)
    w._skip_files.extend(["file1", "file2"])

    # Count how many files actually get processed
    processed_count = 0
    for update in w._run_phc():
        if update["type"] == "update":  # Use "update" not "progress"
            processed_count += 1

    # Should process only 1 file (file3), skip file1 and file2
    assert processed_count == 1


def test_run_qpm_with_files(tmp_path: Path):
    """Test QPM processing with actual files."""
    w = create_widget(tmp_path)
    w._qpm_reconstruct_cbox.setChecked(False)
    w._segment_cbox.setChecked(False)

    # Create valid QPM file
    qpm_file = tmp_path / "test_qpm.tif"
    data = np.zeros((4, 8, 9), dtype=np.float32)
    tifffile.imwrite(qpm_file, data, photometric="minisblack")

    # Process files
    updates = list(w._run_qpm())

    # Should have init and progress updates
    assert len(updates) >= 2
    assert updates[0]["type"] == "init"
    assert updates[0]["total"] == 1

    # Should have at least one progress update
    progress_updates = [u for u in updates if u["type"] == "update"]
    assert len(progress_updates) >= 1


def test_run_phc_with_files(tmp_path: Path):
    """Test PHC processing with actual files."""
    w = create_widget(tmp_path)

    # Create valid PHC file
    phc_file = tmp_path / "test_phc.tif"
    data = np.zeros((15, 16), dtype=np.uint8)
    tifffile.imwrite(phc_file, data, photometric="minisblack")

    # Process files
    updates = list(w._run_phc())

    # Should have init and progress updates
    assert len(updates) >= 2
    assert updates[0]["type"] == "init"
    assert updates[0]["total"] == 1

    # Should have at least one progress update
    progress_updates = [u for u in updates if u["type"] == "update"]
    assert len(progress_updates) >= 1


def test_run_processing_with_directory(tmp_path: Path):
    """Test processing with subdirectories."""
    w = create_widget(tmp_path)
    w._qpm_reconstruct_cbox.setChecked(False)
    w._segment_cbox.setChecked(False)

    # Create subdirectory with files
    subdir = tmp_path / "subdir"
    subdir.mkdir()

    # Create QPM file in subdirectory
    qpm_file = subdir / "test_qpm.tif"
    data = np.zeros((4, 8, 9), dtype=np.float32)
    tifffile.imwrite(qpm_file, data, photometric="minisblack")

    # Process files
    updates = list(w._run_qpm())

    # Should have init and progress updates
    assert len(updates) >= 2
    assert updates[0]["type"] == "init"
    assert updates[0]["total"] == 1  # One file in subdirectory


def test_processing_with_failed_files(tmp_path: Path):
    """Test processing with files that fail validation."""
    w = create_widget(tmp_path)

    # Create invalid QPM file (wrong channels)
    invalid_qpm = tmp_path / "invalid_qpm.tif"
    data = np.zeros((2, 8, 9), dtype=np.float32)  # Only 2 channels instead of 4
    tifffile.imwrite(invalid_qpm, data, photometric="minisblack")

    # Process files
    updates = list(w._run_qpm())

    # Should have validation_errors at the end
    validation_errors = [u for u in updates if u["type"] == "validation_errors"]
    assert len(validation_errors) == 1
    assert "4 channels" in validation_errors[0]["message"]


def test_cancel_during_processing(tmp_path: Path):
    """Test canceling during processing."""
    w = create_widget(tmp_path)
    w._qpm_reconstruct_cbox.setChecked(False)
    w._segment_cbox.setChecked(False)

    # Create multiple files to process
    for i in range(3):
        qpm_file = tmp_path / f"test_qpm_{i}.tif"
        data = np.zeros((4, 8, 9), dtype=np.float32)
        tifffile.imwrite(qpm_file, data, photometric="minisblack")

    # Process with early cancel
    updates = []
    for i, update in enumerate(w._run_qpm()):
        updates.append(update)
        if i == 1:  # Cancel after first update
            w._cancel_requested = True

    # Should have stopped early
    assert len(updates) <= 3  # init + maybe 1-2 updates before canceling


def test_reconstruct_qpm_cancel(tmp_path: Path):
    """Test QPM reconstruction with cancel."""
    w = create_widget(tmp_path)
    w._cancel_requested = True

    # Create test data
    image = np.zeros((4, 8, 9), dtype=np.float32)
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Should return None due to cancel
    result = w._reconstruct_qpm(image, "test", [0, 90, 180, 270], output_dir)
    assert result is None


def test_segment_file_with_cancel(tmp_path: Path):
    """Test segment file with cancel requested."""
    w = create_widget(tmp_path)
    w._cancel_requested = True

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Should return None due to cancel
    image = np.zeros((20, 20), dtype=np.uint8)
    result = w._segment_file(image, "test", output_dir)
    assert result is None


def test_run_widget_methods():
    """Test widget run and cancel methods."""
    w = QPMWidget()

    # Test cancel with no worker
    w.cancel()  # Should not raise error

    # We can't easily test the run() method without mocking create_worker
    # since it starts threads, but we can test that the method exists
    assert hasattr(w, "run")
    assert callable(w.run)
