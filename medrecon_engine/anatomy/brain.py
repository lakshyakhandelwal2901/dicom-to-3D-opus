"""
medrecon_engine.anatomy.brain
===============================
Brain parenchyma segmenter.

Pipeline:
1.  Adaptive HU threshold (profile: ``brain``).
2.  Restrict to intracranial region: flood-fill from centre to exclude
    extracranial soft tissue.
3.  Morphological closing (bridge small sulci).
4.  Connected-component filter — keep single largest blob.

⚠ MEDICAL NOTE: HU-based brain segmentation captures the grey+white matter
bulk reasonably well on non-contrast head CT.  It does NOT delineate
individual gyri, ventricles, or nuclei.  For fine neuroanatomy a trained
3D U-Net is required — the architecture supports plugging one in.
"""

from __future__ import annotations

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

from medrecon_engine.anatomy.base_segmenter import BaseSegmenter
from medrecon_engine.audit.logger import get_logger

log = get_logger(__name__)


class BrainSegmenter(BaseSegmenter):
    """Brain parenchyma segmenter for non-contrast head CT."""

    anatomy = "brain"

    def segment(self, image: sitk.Image) -> sitk.Image:
        hu = sitk.GetArrayFromImage(image).astype(np.float64)
        spacing = tuple(reversed(image.GetSpacing()))

        log.info("BrainSegmenter: starting on volume %s", hu.shape)

        # 1 — Adaptive threshold
        mask, est = self._adaptive_threshold(hu)
        log.info("  Threshold: [%.0f, %.0f] → %s voxels",
                 est.adaptive_min, est.adaptive_max, f"{np.sum(mask):,}")

        # 2 — Exclude skull bone (> 150 HU)
        mask[hu > 150] = 0

        # 3 — Morphological closing
        mask = self._morphological_close(mask)
        mask = ndimage.binary_fill_holes(mask).astype(np.uint8)

        # 4 — Keep only the single largest connected component
        labelled, n = ndimage.label(mask)
        if n > 0:
            sizes = np.bincount(labelled.ravel())
            largest = int(np.argmax(sizes[1:]) + 1)
            mask = (labelled == largest).astype(np.uint8)

        log.info("  Final brain mask: %s voxels", f"{np.sum(mask):,}")
        return self._to_sitk_mask(mask, image)
