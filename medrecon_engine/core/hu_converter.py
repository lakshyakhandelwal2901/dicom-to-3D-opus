"""
medrecon_engine.core.hu_converter
===================================
Convert a raw-pixel SimpleITK image to calibrated Hounsfield Units.

Formula:  HU = pixel_value × RescaleSlope + RescaleIntercept

SimpleITK's ``ImageSeriesReader`` already applies this when the DICOM tags
are present.  This module detects whether conversion has happened and
applies it only if needed, then clips to [HU_CLIP_MIN, HU_CLIP_MAX].

Deterministic — no randomness, no estimation.
"""

from __future__ import annotations

import numpy as np
import SimpleITK as sitk

from medrecon_engine.config.precision_config import PrecisionConfig
from medrecon_engine.audit.logger import get_logger

log = get_logger(__name__)


class HUConverter:
    """Raw pixel → calibrated HU with safety checks."""

    def __init__(self, config: PrecisionConfig | None = None):
        self.cfg = config or PrecisionConfig()

    def convert(
        self,
        image: sitk.Image,
        slope: float = 1.0,
        intercept: float = -1024.0,
    ) -> sitk.Image:
        """Return a new SimpleITK Image whose voxels are in HU.

        Parameters
        ----------
        image : sitk.Image
            The loaded volume (may already be in HU if SimpleITK read the
            DICOM rescale tags).
        slope, intercept : float
            Used only when the volume was manually stacked without rescale.
        """
        arr = sitk.GetArrayFromImage(image).astype(np.float64)

        # Heuristic: if the mean is in raw-pixel territory (0–4095 for 12-bit)
        # the rescale has NOT been applied yet.
        mean_val = float(np.mean(arr))
        if mean_val > 0 and mean_val < 4096:
            log.info("HU converter: applying slope=%.4f intercept=%.1f", slope, intercept)
            arr = arr * slope + intercept
        else:
            log.info("HU converter: volume appears to be in HU already (mean=%.1f)", mean_val)

        # Clip
        arr = np.clip(arr, self.cfg.hu_clip_min, self.cfg.hu_clip_max)

        hu_image = sitk.GetImageFromArray(arr)
        hu_image.CopyInformation(image)

        log.info(
            "HU converter: range [%.0f, %.0f]  mean=%.1f",
            float(np.min(arr)),
            float(np.max(arr)),
            float(np.mean(arr)),
        )
        return hu_image
