"""
medrecon_engine.anatomy.soft_tissue
=====================================
Soft-tissue segmenter (muscle, connective tissue).

Pipeline:
1.  Adaptive HU threshold (profile: ``soft_tissue``).
2.  Morphological closing.
3.  Remove bone and air (subtract those masks if provided, or by HU).
4.  Connected-component filter.
5.  Return binary mask.

⚠ MEDICAL NOTE: Threshold-based soft-tissue segmentation gives
organ-level precision, NOT structure-level.  For fine organ boundaries
(e.g. individual muscles), a 3D U-Net is required.  This architecture
supports plugging one in later.
"""

from __future__ import annotations

import numpy as np
import SimpleITK as sitk

from medrecon_engine.anatomy.base_segmenter import BaseSegmenter
from medrecon_engine.audit.logger import get_logger

log = get_logger(__name__)


class SoftTissueSegmenter(BaseSegmenter):
    """Soft-tissue (muscle / connective) segmenter."""

    anatomy = "soft_tissue"

    def segment(self, image: sitk.Image) -> sitk.Image:
        hu = sitk.GetArrayFromImage(image).astype(np.float64)
        spacing = tuple(reversed(image.GetSpacing()))

        log.info("SoftTissueSegmenter: starting on volume %s", hu.shape)

        # 1 — Adaptive threshold
        mask, est = self._adaptive_threshold(hu)
        log.info("  Threshold: [%.0f, %.0f] → %s voxels",
                 est.adaptive_min, est.adaptive_max, f"{np.sum(mask):,}")

        # 2 — Exclude obvious bone (>300 HU) and air (<-200 HU)
        exclude = (hu > 300) | (hu < -200)
        mask[exclude] = 0

        # 3 — Morphological closing
        mask = self._morphological_close(mask)

        # 4 — Component filter
        mask = self._connected_component_filter(mask, spacing)

        log.info("  Final soft tissue mask: %s voxels", f"{np.sum(mask):,}")
        return self._to_sitk_mask(mask, image)
