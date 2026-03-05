"""
medrecon_engine.anatomy.base_segmenter
========================================
Abstract base class for all anatomy-specific segmenters.

Every segmenter must implement ``segment(image) → sitk.Image``.
The returned image is a **binary mask** (uint8, values 0 or 1) with the
same physical geometry as the input.

The base class provides shared utilities:
- Adaptive HU estimation via ``HUEstimator``.
- Connected-component filtering.
- Morphological closing.
- Fragment-safe minimum-volume filtering.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from skimage.morphology import ball

from medrecon_engine.config.precision_config import PrecisionConfig
from medrecon_engine.hu_model.hu_estimator import HUEstimator, HUEstimationResult
from medrecon_engine.audit.logger import get_logger

log = get_logger(__name__)


class BaseSegmenter(ABC):
    """Abstract segmenter — every anatomy plugin inherits from this."""

    anatomy: str = "unknown"  # override in subclass

    def __init__(self, config: PrecisionConfig | None = None):
        self.cfg = config or PrecisionConfig()
        self.hu_estimator = HUEstimator(self.cfg)

    # ------------------------------------------------------------------ #
    #  Abstract
    # ------------------------------------------------------------------ #
    @abstractmethod
    def segment(self, image: sitk.Image) -> sitk.Image:
        """Segment *image* and return a binary mask (same geometry)."""
        ...

    # ------------------------------------------------------------------ #
    #  Shared building blocks
    # ------------------------------------------------------------------ #
    def _adaptive_threshold(
        self, volume: np.ndarray
    ) -> tuple[np.ndarray, HUEstimationResult]:
        """Threshold *volume* using the adaptive HU estimator.

        Returns ``(binary_mask, estimation_result)``.
        """
        est = self.hu_estimator.estimate(volume, self.anatomy)
        mask = (volume >= est.adaptive_min) & (volume <= est.adaptive_max)
        return mask.astype(np.uint8), est

    def _morphological_close(self, mask: np.ndarray) -> np.ndarray:
        """Binary closing with a ball SE of ``cfg.morphological_closing_radius``."""
        radius = self.cfg.morphological_closing_radius
        if radius <= 0:
            return mask
        struct = ball(radius)
        closed = ndimage.binary_closing(mask, structure=struct, iterations=1)
        closed = ndimage.binary_fill_holes(closed)
        return closed.astype(np.uint8)

    def _connected_component_filter(
        self,
        mask: np.ndarray,
        spacing: tuple[float, float, float],
    ) -> np.ndarray:
        """Keep the top-N largest components above a minimum volume.

        Uses ``cfg.max_retained_components`` and ``cfg.min_component_volume_mm3``.
        """
        labelled, num_features = ndimage.label(mask)
        if num_features == 0:
            return mask

        voxel_vol = float(np.prod(spacing))
        sizes = np.bincount(labelled.ravel())  # idx0 = background

        # Sort component labels by size (descending), skip background
        sorted_labels = np.argsort(sizes[1:])[::-1] + 1  # 1-based labels
        keep_labels: list[int] = []

        for lbl in sorted_labels[: self.cfg.max_retained_components]:
            component_vol = sizes[lbl] * voxel_vol
            if component_vol >= self.cfg.min_component_volume_mm3:
                keep_labels.append(int(lbl))

        out = np.isin(labelled, keep_labels).astype(np.uint8)

        removed = num_features - len(keep_labels)
        if removed > 0:
            log.info(
                "  Fragment filter: kept %d / %d components", len(keep_labels), num_features
            )
        return out

    def _to_sitk_mask(self, mask: np.ndarray, reference: sitk.Image) -> sitk.Image:
        """Wrap a numpy mask back into a SimpleITK image with original geometry."""
        sitk_mask = sitk.GetImageFromArray(mask.astype(np.uint8))
        sitk_mask.CopyInformation(reference)
        return sitk_mask
