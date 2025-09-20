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


def test_get_total_number_of_files(tmp_path: Path):
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
    total = w._get_total_number_of_files()
    assert total == 3


def test_process_qpm_tif_success_and_failure(tmp_path: Path):
    # valid qpm tif (4, H, W)
    valid = tmp_path / "valid_qpm.tif"
    data = np.zeros((4, 8, 9), dtype=np.float32)
    tifffile.imwrite(valid, data, photometric="minisblack")

    w = create_widget(tmp_path)
    # disable heavy operations
    w._qpm_reconstruct_cbox.setChecked(False)
    w._segment_cbox.setChecked(False)

    ok, name = w._process_qpm_tif(valid, [0, 90, 180, 270])
    assert ok is True
    assert name == ""  # _process_qpm_tif returns empty string on success
    outdir = tmp_path / f"{valid.stem}{QPM_PROCESSED}"
    assert outdir.exists() and outdir.is_dir()

    # invalid qpm tif (wrong channels)
    invalid = tmp_path / "invalid_qpm.tif"
    tifffile.imwrite(
        invalid, np.zeros((2, 5, 5), dtype=np.float32), photometric="minisblack"
    )
    ok, msg = w._process_qpm_tif(invalid, [0, 90, 180, 270])
    assert ok is False
    assert invalid.stem in msg


def test_process_phc_tif_success_and_failure(tmp_path: Path):
    # valid phc tif (2D)
    valid = tmp_path / "valid_phc.tif"
    tifffile.imwrite(
        valid, np.zeros((15, 16), dtype=np.uint8), photometric="minisblack"
    )

    w = create_widget(tmp_path)
    ok, name = w._process_phc_tif(valid)
    assert ok is True
    assert name == ""  # _process_phc_tif returns empty string on success
    outdir = tmp_path / f"{valid.stem}{PHC_PROCESSED}"
    assert outdir.exists() and outdir.is_dir()

    # invalid (3D)
    bad = tmp_path / "bad_phc.tif"
    tifffile.imwrite(bad, np.zeros((3, 5, 5), dtype=np.uint8), photometric="minisblack")
    ok, msg = w._process_phc_tif(bad)
    assert ok is False
    assert bad.stem in msg


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
