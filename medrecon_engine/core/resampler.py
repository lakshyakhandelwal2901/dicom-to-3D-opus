"""
medrecon_engine.core.resampler
================================
Resample a SimpleITK image to isotropic spacing, respecting the
precision bounds in ``PrecisionConfig``.

This is a separate, auditable step so the audit log records both the
original and the resampled spacing.
"""

from __future__ import annotations

import numpy as np
import SimpleITK as sitk

from medrecon_engine.config.precision_config import PrecisionConfig
from medrecon_engine.audit.logger import get_logger

log = get_logger(__name__)


class Resampler:
    """Resample a volume to isotropic target spacing."""

    INTERP_MAP = {
        "linear": sitk.sitkLinear,
        "bspline": sitk.sitkBSpline,
        "nearest": sitk.sitkNearestNeighbor,
    }

    def __init__(self, config: PrecisionConfig | None = None):
        self.cfg = config or PrecisionConfig()

    def resample(self, image: sitk.Image) -> sitk.Image:
        """Resample *image* to ``self.cfg.resampling_target`` spacing.

        Returns a new SimpleITK image — the original is not mutated.
        """
        original_spacing = image.GetSpacing()
        original_size = image.GetSize()
        target_spacing = self.cfg.resampling_target

        # Compute new grid size
        new_size = [
            int(round(osz * ospc / tspc))
            for osz, ospc, tspc in zip(original_size, original_spacing, target_spacing)
        ]

        interpolator = self.INTERP_MAP.get(
            self.cfg.resampling_interpolation, sitk.sitkLinear
        )

        # Default pixel = minimum HU in the volume (usually air / padding)
        arr_view = sitk.GetArrayViewFromImage(image)
        default_value = float(np.min(arr_view))

        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(target_spacing)
        resampler.SetSize(new_size)
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetTransform(sitk.Transform())
        resampler.SetDefaultPixelValue(default_value)
        resampler.SetInterpolator(interpolator)

        resampled = resampler.Execute(image)

        log.info(
            "Resampler: %s sp=(%.3f,%.3f,%.3f) → %s sp=(%.2f,%.2f,%.2f)",
            original_size, *original_spacing,
            resampled.GetSize(), *resampled.GetSpacing(),
        )
        return resampled
