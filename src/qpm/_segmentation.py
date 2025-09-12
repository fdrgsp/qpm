from __future__ import annotations
from cellpose import models, core, io
import numpy as np


class CellposeSAMSegmentation:
    """A class to handle cellpose segmentation of bacteria images."""

    def __init__(self) -> None:
        io.logger_setup()

        self._model = models.CellposeModel(gpu=core.use_gpu(), pretrained_model="cpsam")

        # parameters (maybe expose them later on)
        self.eval_flow_threshold = 0.4  # default is 0.4
        self.eval_cellprob_threshold = 0.0  # default is 0.0
        self.eval_tile_norm_blocksize = 0  # default is 0
        self.eval_batch_size = 8  # default is 8

    def eval(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run CellposeSAM segmentation on the given image."""
        labels, flows, _ = self._model.eval(
            image,
            batch_size=self.eval_batch_size,
            flow_threshold=self.eval_flow_threshold,
            cellprob_threshold=self.eval_cellprob_threshold,
            normalize={"tile_norm_blocksize": self.eval_tile_norm_blocksize},
        )
        return labels, flows
