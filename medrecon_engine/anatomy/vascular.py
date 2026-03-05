"""
medrecon_engine.anatomy.vascular
==================================
Vascular segmenter — aorta, iliac, hepatic, renal arteries, etc.

Pipeline:
1.  Adaptive HU threshold (profile: ``vascular``).
    — The HU estimator automatically shifts for contrast-enhanced scans
      because the vascular profile has ``contrast_shift=150``.
2.  Morphological closing.
3.  Connected-component filter (keep top-5 for branching vasculature).

⚠ MEDICAL NOTE: HU-threshold vascular segmentation works ONLY on
contrast-enhanced CT angiography (CTA).  On non-contrast scans, vessels
overlap with soft tissue and the result will be noisy.  True vascular
segmentation requires AI.  This module gives a reasonable starting point
for CTA data.
"""

from __future__ import annotations

import numpy as np
import SimpleITK as sitk

from medrecon_engine.anatomy.base_segmenter import BaseSegmenter
from medrecon_engine.audit.logger import get_logger

log = get_logger(__name__)


class VascularSegmenter(BaseSegmenter):
    """Vascular tree segmenter (best on contrast-enhanced CTA)."""

    anatomy = "vascular"

    def segment(self, image: sitk.Image) -> sitk.Image:
        hu = sitk.GetArrayFromImage(image).astype(np.float64)
        spacing = tuple(reversed(image.GetSpacing()))

        log.info("VascularSegmenter: starting on volume %s", hu.shape)

        # 1 — Adaptive threshold
        mask, est = self._adaptive_threshold(hu)
        log.info("  Threshold: [%.0f, %.0f] → %s voxels",
                 est.adaptive_min, est.adaptive_max, f"{np.sum(mask):,}")

        # 2 — Exclude definite bone (>700 HU) which overlaps upper range
        mask[hu > 700] = 0

        # 3 — Morphological closing
        mask = self._morphological_close(mask)

        # 4 — Component filter (vessels branch → allow top 5)
        mask = self._connected_component_filter(mask, spacing)

        log.info("  Final vascular mask: %s voxels", f"{np.sum(mask):,}")
        return self._to_sitk_mask(mask, image)
