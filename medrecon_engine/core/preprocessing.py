"""
medrecon_engine.core.preprocessing
====================================
Pre-segmentation volume conditioning.

Steps (all optional, controlled by PrecisionConfig):
1.  Anisotropic Gaussian smoothing to reduce noise without blurring
    across slice boundaries.
2.  Volume cropping to the non-air region to speed up downstream
    marching cubes (optional optimisation — does NOT discard anatomy).

This module operates on numpy arrays (extracted from SimpleITK images)
rather than SimpleITK filters, giving us full control and auditability.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from medrecon_engine.config.precision_config import PrecisionConfig
from medrecon_engine.audit.logger import get_logger

log = get_logger(__name__)


class Preprocessor:
    """Volume conditioning before segmentation."""

    def __init__(self, config: PrecisionConfig | None = None):
        self.cfg = config or PrecisionConfig()

    # ------------------------------------------------------------------ #
    #  Gaussian smoothing
    # ------------------------------------------------------------------ #
    def smooth(
        self,
        volume: np.ndarray,
        spacing: tuple[float, float, float],
    ) -> np.ndarray:
        """Anisotropic Gaussian smoothing.

        The sigma in each axis is ``cfg.gaussian_sigma_mm / spacing[axis]``
        so smoothing is physically isotropic regardless of voxel size.
        """
        sigma_mm = self.cfg.gaussian_sigma_mm
        if sigma_mm <= 0:
            return volume

        # spacing is (z, y, x) — volume axes follow the same order
        sigma_voxels = tuple(sigma_mm / s for s in spacing)
        smoothed = ndimage.gaussian_filter(volume, sigma=sigma_voxels)

        log.info(
            "Preprocessor: Gaussian σ=%.2f mm → voxel σ=(%.2f, %.2f, %.2f)",
            sigma_mm, *sigma_voxels,
        )
        return smoothed

    # ------------------------------------------------------------------ #
    #  Crop to non-air bounding box (optimisation)
    # ------------------------------------------------------------------ #
    @staticmethod
    def crop_to_body(
        volume: np.ndarray,
        hu_air_threshold: float = -500.0,
        margin: int = 10,
    ) -> tuple[np.ndarray, tuple[slice, slice, slice]]:
        """Crop volume to the bounding box of non-air voxels.

        Returns the cropped array AND the slicing tuple so we can map
        coordinates back to the original grid later.
        """
        body_mask = volume > hu_air_threshold
        if not np.any(body_mask):
            log.warning("Preprocessor: entire volume is below air threshold; no crop applied.")
            full_slices = (
                slice(0, volume.shape[0]),
                slice(0, volume.shape[1]),
                slice(0, volume.shape[2]),
            )
            return volume, full_slices

        coords = np.argwhere(body_mask)
        lo = np.maximum(coords.min(axis=0) - margin, 0)
        hi = np.minimum(coords.max(axis=0) + margin + 1, volume.shape)

        slicing = (
            slice(int(lo[0]), int(hi[0])),
            slice(int(lo[1]), int(hi[1])),
            slice(int(lo[2]), int(hi[2])),
        )

        cropped = volume[slicing]
        log.info(
            "Preprocessor: cropped %s → %s  (saved %.1f%% voxels)",
            volume.shape,
            cropped.shape,
            (1 - cropped.size / volume.size) * 100,
        )
        return cropped, slicing
