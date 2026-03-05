"""
medrecon_engine.anatomy.lung
==============================
Lung parenchyma segmenter.

Pipeline:
1.  Adaptive HU threshold (profile: ``lung``).
2.  Large morphological closing (lungs have airways / vessels inside).
3.  Fill holes (the mediastinum creates a gap between left/right).
4.  Keep only the 2 largest components (left + right lung).
5.  Return binary mask.

NOTE: This segments the *parenchyma* (air-filled tissue), not airways.
"""

from __future__ import annotations

import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from skimage.morphology import ball

from medrecon_engine.anatomy.base_segmenter import BaseSegmenter
from medrecon_engine.audit.logger import get_logger

log = get_logger(__name__)


class LungSegmenter(BaseSegmenter):
    """Lung parenchyma segmenter."""

    anatomy = "lung"

    # Lungs need bigger closing to bridge internal vessels/airways
    LUNG_CLOSING_RADIUS: int = 5
    # We expect exactly 2 lungs
    MAX_LUNG_COMPONENTS: int = 2
    MIN_LUNG_VOLUME_MM3: float = 50_000.0  # ~50 cm³ — tiny lungs still count

    def segment(self, image: sitk.Image) -> sitk.Image:
        hu = sitk.GetArrayFromImage(image).astype(np.float64)
        spacing = tuple(reversed(image.GetSpacing()))

        log.info("LungSegmenter: starting on volume %s", hu.shape)

        # 1 — Adaptive threshold (negative HU range)
        mask, est = self._adaptive_threshold(hu)
        log.info("  Threshold: [%.0f, %.0f] → %s voxels",
                 est.adaptive_min, est.adaptive_max, f"{np.sum(mask):,}")

        # 2 — Aggressive closing
        struct = ball(self.LUNG_CLOSING_RADIUS)
        mask = ndimage.binary_closing(mask, structure=struct, iterations=1).astype(np.uint8)

        # 3 — Fill holes
        mask = ndimage.binary_fill_holes(mask).astype(np.uint8)

        # 4 — Keep top-2 components above volume threshold
        labelled, n = ndimage.label(mask)
        if n > 0:
            voxel_vol = float(np.prod(spacing))
            sizes = np.bincount(labelled.ravel())
            sorted_labels = np.argsort(sizes[1:])[::-1] + 1
            keep = []
            for lbl in sorted_labels[:self.MAX_LUNG_COMPONENTS]:
                if sizes[lbl] * voxel_vol >= self.MIN_LUNG_VOLUME_MM3:
                    keep.append(int(lbl))
            mask = np.isin(labelled, keep).astype(np.uint8)

        log.info("  Final lung mask: %s voxels", f"{np.sum(mask):,}")
        return self._to_sitk_mask(mask, image)
