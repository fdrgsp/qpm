from __future__ import annotations
from cellpose import models, core, io
import numpy as np


class CellposeSAMSegmentation:
    """A class to handle cellpose segmentation of bacteria images."""

    def __init__(self) -> None:
        io.logger_setup()

        self._model = models.CellposeModel(gpu=core.use_gpu(), pretrained_model="cpsam")

        # parameters (maybe expose them later on)
        self.eval_flow_threshold: float = 0.4  # default is 0.4
        self.eval_cellprob_threshold: float = 0.0  # default is 0.0
        self.eval_tile_norm_blocksize: int = 0  # default is 0
        self.eval_batch_size: int = 8  # default is 8
        self.max_size_fraction: float = 0.4  # default is 0.4
        self.min_size: int = 15  # default is 15
        self.diameter: int | None = None  # default is None

    def set_parameters(
        self,
        *,
        flow_threshold: float = 0.4,
        cellprob_threshold: float = 0.0,
        min_size: int = 15,
        max_size_fraction: float = 0.4,
        tile_norm_blocksize: int = 0,
        batch_size: int = 8,
        diameter: int | None = None,
    ) -> None:
        """Set parameters for CellposeSAM segmentation."""
        self.eval_flow_threshold = flow_threshold
        self.eval_cellprob_threshold = cellprob_threshold
        self.eval_tile_norm_blocksize = tile_norm_blocksize
        self.eval_batch_size = batch_size
        self.max_size_fraction = max_size_fraction
        self.min_size = min_size
        self.diameter = diameter or None

    def eval(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run CellposeSAM segmentation on the given image."""
        labels, flows, _ = self._model.eval(
            image,
            diameter=self.diameter,
            batch_size=self.eval_batch_size,
            flow_threshold=self.eval_flow_threshold,
            cellprob_threshold=self.eval_cellprob_threshold,
            max_size_fraction=self.max_size_fraction,
            min_size=self.min_size,
            normalize={"tile_norm_blocksize": self.eval_tile_norm_blocksize},
        )
        return labels, flows
