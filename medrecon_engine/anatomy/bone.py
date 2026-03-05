"""
medrecon_engine.anatomy.bone
==============================
Bone segmenter — cortical + cancellous.

Pipeline:
1.  Adaptive HU threshold (profile: ``bone``).
2.  Morphological closing (fill small gaps in cortex).
3.  Connected-component filter (keep top-5, remove dust).
4.  Return binary mask.

This handles: pelvis, femur, tibia, spine, skull, ribs, etc.
The adaptive HU system adjusts per-patient automatically.
"""

from __future__ import annotations

import numpy as np
import SimpleITK as sitk

from medrecon_engine.anatomy.base_segmenter import BaseSegmenter
from medrecon_engine.audit.logger import get_logger

log = get_logger(__name__)


class BoneSegmenter(BaseSegmenter):
    """Universal bone segmenter (cortical + cancellous)."""

    anatomy = "bone"

    def segment(self, image: sitk.Image) -> sitk.Image:
        """Segment bone from a HU-calibrated CT volume.

        Parameters
        ----------
        image : sitk.Image
            HU-calibrated, resampled volume.

        Returns
        -------
        sitk.Image
            Binary mask (uint8, 0/1) with same geometry.
        """
        hu = sitk.GetArrayFromImage(image).astype(np.float64)
        spacing = tuple(reversed(image.GetSpacing()))  # (z, y, x)

        log.info("BoneSegmenter: starting on volume %s", hu.shape)

        # 1 — Adaptive threshold
        mask, est = self._adaptive_threshold(hu)
        log.info("  Threshold: [%.0f, %.0f] → %s voxels",
                 est.adaptive_min, est.adaptive_max, f"{np.sum(mask):,}")

        # 2 — Morphological closing
        mask = self._morphological_close(mask)

        # 3 — Fragment-safe component filter
        mask = self._connected_component_filter(mask, spacing)

        log.info("  Final bone mask: %s voxels", f"{np.sum(mask):,}")
        return self._to_sitk_mask(mask, image)
